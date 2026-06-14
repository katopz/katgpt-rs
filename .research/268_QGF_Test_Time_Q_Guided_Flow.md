# Research 268: QGF — Test-Time Q-Guided Flow for Modelless Inference

**Paper:** [arXiv:2606.11087](https://arxiv.org/pdf/2606.11087) — Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning (Zhou, Peng, Xu, Li, Springenberg, Frans, Levine; UC Berkeley + Physical Intelligence, Jun 2026)
**Date:** 2026-06-14, distilled 2026-06-14
**Domain:** katgpt-rs (open, MIT) — modelless inference engine
**Related Research:** 150 (RecFM Recursive Flow Matching), 175 (ThoughtFold), 204 (NFCoT Normalizing Flow CoT), 215 (ECHO Env Prediction), 217 (TRD Trajectory Refined), 229 (ProgramAsWeights), 267 (Thicket Variance Probe), 003 (Commercial Strategy)
**Related Plans:** 229 (NFCoT FlowScore — MARGINAL, QGF unblocks), 195 (ThoughtFold), 217 (NextLat Belief Drafter), 247 (ECHO Predictor), 262 (Latent Physics Primitives), 267 (Thicket)
**Verdict:** 🟢 **GAIN** — QGF is the missing principled gradient signal that turns NFCoT FlowScore from marginal to GOAT. Four modelless fusion ideas; F1 (QGuidedDrafter) + F2 (FirstOrderProjector) ship immediately. Engine/fuel split is clean: primitive is public, critic weights + schedules are private.

---

## TL;DR

QGF (Q-Guided Flow) is a **test-time RL** algorithm: train a policy with stable supervised behavioral cloning (BC) + train a critic (Q-function) separately with IQL, then at **inference time** use the critic gradient to steer the policy's denoising process toward higher-value actions — **without any policy training**. The key trick: evaluate the critic gradient at a **first-order Euler approximation of the clean output** (`â₁ = aₜ + (1-t)·v_θ(s,aₜ,t)`) and **drop the Jacobian entirely** (J ≈ I). This avoids both the OOD bias of gradient-at-noisy-action and the high variance / cost of backprop-through-time.

**Why we care:** QGF's principle generalizes far beyond continuous diffusion/flow policies. It says: *at any iterative generation step, evaluate your value-gradient oracle at a one-step projection of the final output, not at the current intermediate, and use it as a velocity bias.* We already have every piece of this in `katgpt-core`:

| QGF Component | Our Existing Analog |
|---|---|
| Reference policy `π̂` (BC flow) | DDTree marginals + NFCoT FlowScore (Plan 229) |
| Critic `Q(s,a)` (IQL) | `LeoHead::all_goals_q()` + `FlowFieldCache` |
| Denoising step `v_θ` | `SpeculativeGenerator::generate()` |
| Gradient `∇_a Q(s,â₁)` | `FlowField::gradient()` + `ActionBridge` dot product |
| One-step projection `â₁` | `precision_aware_draft.rs` / NextLat belief drafter (Plan 217) |
| Guidance weight `1/β` | Existing bandit/pruner thresholds (sigmoid-gated) |

**Verdict by 003 strategy:** The QGF *mechanism* (gradient-guided speculative generation, first-order projection, drop-Jacobian estimator) is **generic engine infrastructure → public MIT katgpt-rs**. The trained IQL critic weights, BC reference policy, guidance-weight schedules, and game-domain configs are **fuel → private riir-ai**. This is the canonical engine/fuel split.

---

## 1. Paper Core (Distilled in Our Terms)

### 1.1 The Problem QGF Solves

Offline RL with expressive policies (diffusion / flow matching) is unstable because the actor is trained to maximize a *changing* critic via backprop through a multi-step denoising process. This couples actor and critic training, requires tuning a behavioral-constraint weight, and scales poorly.

### 1.2 The QGF Insight

**Decouple completely.** Train:
1. A **reference policy** `π̂` with standard supervised BC (stable, scalable).
2. A **critic** `Q(s,a)` with in-sample value learning (IQL — no policy sampling needed).

Then at **test time only**, guide the reference policy's denoising with the critic gradient. No policy gradient, no actor-critic instability, no behavioral-constraint tuning during training.

### 1.3 The Closed-Form Target

The KL-regularized objective
```
J(π) = E_π[Σ γ^t r(s_t,a_t)] − β · E_s[D_KL(π(·|s) ‖ π̂(·|s))]
```
has closed-form solution:
```
π(a|s) ∝ π̂(a|s) · exp(Q(s,a))^(1/β)
```
Taking the score (gradient of log-density):
```
∇_a log π(a|s) = ∇_a log π̂(a|s) + (1/β) · ∇_a Q(s,a)
```
So the improved policy's score = reference score + scaled critic gradient. This is **classifier guidance with Q replacing the classifier**.

### 1.4 The Three Gradient Estimators (and why QGF wins)

| Estimator | Formula | Cost | Variance | Bias | Result |
|---|---|---|---|---|---|
| **OOD** (QFQL [24]) | `∇_{a_t} Q(s, a_t)` | Cheap | Low | **Biased** (critic untrained on noisy actions) | Suboptimal — exploits Q off-manifold |
| **BPTT** (DQL [63]) | `∇_{a_t} Q(s, ODE(a_t))` | Expensive | **High** | Low | Unstable, sensitive to noise |
| **QGF** (ours) | `∇_{â_1} Q(s, â_1)` where `â_1 = a_t + (1-t)·v_θ(s,a_t,t)`, **J ≈ I** | Cheap | **Lowest** | Low | **Best** |

**The two counterintuitive wins:**
1. **First-order approximation > full denoising chain.** Using one big Euler step `â_1 = a_t + (1-t)·v_θ` instead of the full ODE integration gives *better* results because it allows mode selection (deviation from exact dataset distribution), not just mode coverage.
2. **Dropping the Jacobian > including it.** Setting `J = ∂â_1/∂a_t ≈ I` instead of computing the true Jacobian gives *lower variance* and better Q-optimization. The Jacobian is ill-behaved at early steps where the Euler approximation is crude.

### 1.5 Algorithm (Inference-Only)

```
Input: state s, reference flow v_θ, critic Q, guidance weight 1/β, step δ=1/T
a_0 ~ N(0, I)
for t = 0, δ, 2δ, ..., 1−δ:
    â_1 ← a_t + (1−t) · v_θ(s, a_t, t)        # one-step projection (FIRST-ORDER)
    g   ← ∇_{â_1} Q(s, â_1)                    # critic gradient at projection (DROP JACOBIAN)
    a_{t+δ} ← a_t + δ · (v_θ(s, a_t, t) + (1/β) · g)   # guided Euler step
return a_1
```

### 1.6 Key Empirical Results

- **Beats all test-time methods** (BFN, GradStep, QFQL, BPTT, CFGRL, RobustQ) on 20 OGBench tasks.
- **Competitive with best training-time methods** (EDP, QAM, FQL) — sometimes better — without policy RL training.
- **Scales better with model size**: 825K → 3.2M params gives QGF ~4× jump; QAM (training-time) does not improve.
- **Cheap**: orders of magnitude fewer FLOPs than BFN(N=16) for matched quality.
- **Critic-agnostic**: works with IQL or QAM-bootstrapped critics; better critic → better QGF.

---

## 2. Why This Lands in katgpt-rs (Modelless, Public)

Per the 003 verdict strategy:

| Aspect | Classification | Reasoning |
|---|---|---|
| The QGF *algorithm* (gradient-guided denoising) | **Public (katgpt-rs)** | Generic inference primitive — the *what*, not the *how*. Same category as DDTree, speculative decoding, ConstraintPruner trait. |
| First-order projection + drop-Jacobian estimator | **Public (katgpt-rs)** | Mathematical technique, no moat risk. Adoption value. |
| The Q-gradient oracle *trait* | **Public (katgpt-rs)** | Generic interface, like `SpeculativeGenerator` and `LeoHead`. |
| Which specific critic (IQL vs QAM) we use for game X | **Private (riir-ai)** | Naming the technique hands competitors the direction. |
| Guidance-weight `1/β` schedules per game/domain | **Private (riir-ai)** | The HOW — tuned configs are the fuel. |
| Trained Q-critic weights (`.bin`) | **Private (riir-ai)** | GPU-hours to produce. Never shipped. |
| BC reference policy LoRA weights | **Private (riir-ai)** | Already private per existing policy. |

**The primitive is the engine. The weights and recipes are the fuel.** QGF fits the split perfectly — same as NFCoT FlowScore (Plan 229) and the LeoHead trait.

---

## 3. Fusion Ideas for katgpt-rs (Modelless Extractions)

### Fusion Idea 1: QGuidedDrafter — SpeculativeGenerator with Q-Gradient Velocity Bias ⭐ GOAT

**What:** Extend the `SpeculativeGenerator` trait with an optional Q-gradient guidance term, applied as a velocity bias during candidate generation. This is the direct modelless port of QGF's Algorithm 1.

**Current state:** `SpeculativeGenerator::generate()` produces candidates from DDTree marginals. NFCoT FlowScore (Plan 229) *scores* candidates post-hoc by flow density but does not *steer* generation. Selection is greedy on score.

**QGF fusion:** During generation step `t`, compute a one-step projection of the current chain prefix to its likely final output, query the Q-gradient oracle at that projection, and add `(1/β) · g` to the generation velocity (logit bias / marginal tilt).

**Concrete discrete analogue:**
```rust
// At DDTree depth t, with current marginal p_t and drafter velocity v_t:
let projected_final = project_one_step(prefix, v_t, remaining_depth);  // â_1
let q_grad = q_oracle.gradient(state, &projected_final);               // ∇_{â_1} Q
let guided_marginal = combine(p_t, q_grad, guidance_weight);           // p_t + (1/β)·g
```

**Why it's not a direct mapping:** QGF operates on continuous action spaces with a flow-matching velocity field. We adapt the *structural principle* (gradient at projection, not at intermediate) to **discrete token/action marginals**. The "velocity" becomes a logit bias; the "Euler step" becomes a k-step lookahead projection using the existing drafter; the "Jacobian drop" becomes "use the gradient directly without chain-rule backprop through the drafter."

**Alignment with optimization.md:**
- O(vocab_size) per position for the gradient (same as softmax) — SIMD-friendly
- One extra drafter call per guidance step for the projection (amortized via existing `generate_batch`)
- Zero allocation: reuse existing marginal buffers
- Pre-compute guidance weight `1/β` once per query (bandit-adaptive)

**Unblocks:** NFCoT FlowScore (Plan 229) was MARGINAL because post-hoc scoring doesn't improve generation, only selection. QGF *steers* generation, which is the missing piece.

**Verdict by 003 strategy:** Generic engine primitive, MIT public. Direct gain, ship it.

---

### Fusion Idea 2: FirstOrderProjector — One-Step Chain Projection ⭐ GOAT

**What:** A reusable function that takes a partial generation chain (DDTree prefix, CoT prefix, action chunk prefix) and projects it to its likely final output using a single "big Euler step" — calling the drafter once with the remaining budget collapsed to one step.

**Current state:** NextLat belief drafter (Plan 217) does k-step belief-state prediction. precision_aware_draft.rs does single-step lookahead. But neither exposes a clean "project to final" primitive.

**QGF fusion:** Extract the *structural primitive* — `project_to_final(prefix, drafter, remaining_steps) -> projected_final` — as a standalone, zero-cost function. This is QGF's `â_1 = a_t + (1-t)·v_θ` generalized to any generative process.

**Why it matters:** The first-order projection is the **load-bearing insight** of QGF. It's what makes the gradient cheap (no BPTT) and low-variance (no Jacobian). By exposing it as a primitive, every downstream consumer (NFCoT, ThoughtFold, ECHO, TRD) can query "what would this chain likely end up as?" in O(1) drafter calls.

**Alignment with optimization.md:**
- Single drafter forward pass per projection — amortized in batches
- Reuses existing `SpeculativeGenerator::generate_batch`
- No new allocations — writes into caller-provided buffer

**Verdict by 003 strategy:** Pure modelless primitive. Ship it.

---

### Fusion Idea 3: DropJacobianQGradient — Variance-Reduced Gradient Estimator

**What:** A `QGradientOracle` trait that returns the critic gradient at a projected output, with the Jacobian explicitly set to identity. Document *why* (variance reduction) so future contributors don't "fix" it.

**Current state:** `FlowField::gradient()` computes a spatial gradient from Q-values via FFT smoothing. `ActionBridge::select_action` does a dot-product + sigmoid. Neither exposes a clean "gradient of Q w.r.t. action" interface.

**QGF fusion:** Generalize the existing gradient computation into a trait:
```rust
pub trait QGradientOracle {
    type Action;
    /// ∇_a Q(s, a) evaluated at the projected action.
    /// NOTE: Jacobian is intentionally dropped (J ≈ I) per QGF paper §5.
    /// Do NOT add chain-rule backprop — it increases variance (see Fig 3).
    fn q_gradient_at(&self, state: &Self::State, projected_action: &Self::Action) -> Vec<f32>;
}
```

**Why it matters:** This codifies QGF's counterintuitive finding (drop the Jacobian) as a *documented design decision*, preventing well-meaning contributors from re-introducing the high-variance BPTT path. The FFT smoothing in `FlowFieldCache` is already a form of variance reduction — QGF explains *why* it works.

**Alignment with optimization.md:**
- Trait is zero-cost abstraction
- Implementations reuse existing SIMD dot-product kernels
- No new allocations — gradient written into caller buffer

**Verdict by 003 strategy:** Public trait, generic interface. Ship it.

---

### Fusion Idea 4: VarianceAdaptiveGuidance — Sigmoid-Gated Guidance Weight

**What:** Adapt the guidance weight `1/β` per-query based on the critic's confidence (variance of Q across the action manifold). High-confidence critic → strong guidance; low-confidence → fall back to pure BC reference.

**Current state:** BanditPruner and various threshold gates adapt compute per-query. Thicket (Plan 267) does variance-probe routing.

**QGF fusion:** The paper shows (Fig 20) that guidance weight has a sweet spot — too low = no improvement, too high = off-manifold exploitation. We make this adaptive:
```
guidance_weight = sigmoid(k · (confidence − threshold))
where confidence = 1 − normalized_variance(Q(s, ·))
```

This is **sigmoid, not softmax** (per project rules), and it naturally routes:
- **Low confidence → pure BC reference** (fallback, safe)
- **High confidence → strong Q-guidance** (aggressive, high-quality)

**Why it's not a direct mapping:** The paper uses a fixed `1/β` tuned per-domain. We make it *adaptive per-query* using the critic's own variance as the signal — a novel extension that the paper doesn't explore but is natural given our existing variance-probe infrastructure (Thicket).

**Alignment with optimization.md:**
- Variance computed once per state (already done in FlowFieldCache)
- Sigmoid is one SIMD op
- Threshold is bandit-adaptive (reuse existing PrudentBanker infrastructure)

**Verdict by 003 strategy:** Modelless adaptive compute. Ship it, but **gate it** — adaptive guidance is subtle and needs real-world validation (same verdict as Recursive SpecHop in Research 150).

---

## 4. GOAT Verdict Summary

| Fusion Idea | Target | Gain Mechanism | Perf Impact | Verdict |
|---|---|---|---|---|
| **F1: QGuidedDrafter** | SpeculativeGenerator + NFCoT | Q-gradient velocity bias during generation | 1 extra drafter call per guidance step (amortized) | ✅ Ship modelless |
| **F2: FirstOrderProjector** | Reusable primitive | One-step chain projection for all consumers | O(1) drafter calls, zero alloc | ✅ Ship modelless |
| **F3: DropJacobianQGradient** | QGradientOracle trait | Variance-reduced gradient estimator (documented) | Zero-cost trait | ✅ Ship modelless |
| **F4: VarianceAdaptiveGuidance** | Sigmoid-gated 1/β | Per-query adaptive guidance weight | Negligible (sigmoid + variance reuse) | ⚠️ Keep gated — needs validation |

**Composite verdict:** F1+F2+F3 form a coherent primitive that **unblocks NFCoT FlowScore (Plan 229)** from MARGINAL to potential GOAT. F4 is the adaptive layer on top.

---

## 5. CPU / GPU / ANE Auto-Route (per constraints)

The QGF primitive has three compute kernels, each with a natural substrate:

| Kernel | CPU (SIMD) | GPU | ANE |
|---|---|---|---|
| First-order projection (drafter call) | ✅ Default for small batches | ✅ Batch ≥ 8 | ✅ NPC brain (existing `npc_ane_backend`) |
| Q-gradient at projection (dot product + sigmoid) | ✅ **Always CPU** — NEON/AVX2, < 1μs | ○ Only for huge action spaces | ✅ Critic forward on ANE |
| Guidance-weight sigmoid | ✅ Always CPU (scalar) | ✗ Overkill | ✗ Overkill |

**Threshold-based routing (per constraint 9):**
```
if action_space_size < 1024:
    route = CPU_SIMD              # dot product is fastest here
elif batch_size >= 8 and action_space_size >= 1024:
    route = GPU_BATCH             # amortize dispatch overhead
else:
    route = CPU_SIMD              # default safe path
```

The Q-critic *forward pass* (computing Q-values) routes to ANE for NPC brains (existing infrastructure) or GPU for large批 training.

---

## 6. Plasma / Hot / Warm / Cold / Freeze Path (per constraint 8)

QGF's Q-gradient oracle naturally spans the five tiers:

| Tier | What lives here | Latency | QGF use |
|---|---|---|---|
| **Plasma** (ternary SIMD) | Compressed critic as ternary direction vectors (`ActionBridge` i8 directions) | < 100ns | Default game-time guidance — dot product of f32 Q-values with i8 directions |
| **Hot** (< 1μs) | Full f32 Q-values in L1/L2 (`LeoHead::all_goals_q` cached) | < 1μs | Per-frame guidance for active NPCs |
| **Warm** (GPU) | Full Q-critic forward pass | ~1ms (batched) | Batch guidance for many NPCs / training |
| **Cold** (Turso, encrypted) | Q-table snapshots per zone/episode | ~10ms (load) | Episode-end consolidation, cross-session skill transfer |
| **Freeze** (anyrag fallback) | BC reference policy only, no guidance | N/A | Engine always boots — graceful degradation when critic unavailable |

**Bridge functions (per latent-vs-raw rules):**
- Raw `MapPos { x, y }` → raw, synced → feeds Q-critic as input
- Q-values (latent-ish, scalar projections) → synced as the 5 scalar emotion outputs, NOT the full vector
- Q-gradient → local, not synced — applied per-entity at inference time only

This is consistent with the existing `ActionBridge` which already bridges latent Q-vectors to raw actions via sigmoid-gated projection.

---

## 7. What NOT to Apply

**QGF's continuous flow-matching training** (the `v_θ` velocity field trained with the flow matching loss in Eq. 2) is model-based. We do NOT implement continuous diffusion/flow policies in katgpt-rs — same verdict as Research 079 (ELF), 041 (RePlaid), 010 (ColaDLM): incompatible with our discrete DDTree inference path. The modelless extractions above capture the *principle* without the continuous machinery.

**QGF's IQL critic training** (TD loss with expectile regression) is model-based. That belongs in riir-ai (see companion doc `riir-ai/.research/123_QGF_Critic_Training_Verdict.md`). We only consume the trained critic via the `LeoHead` trait, which already exists.

**QGF's OGBench manipulation experiments** are robotics-specific. We validate on our own arenas (Bomber, Go, FFT, Sudoku, Rust syntax validation).

**Backprop-through-time (BPTT) variant** is explicitly worse — do not implement. Document this in the trait docstring so it's not re-introduced.

---

## 8. Relationship to Existing Research

| Research | Overlap | Distinction |
|---|---|---|
| **150 RecFM** (Recursive Flow Matching) | Both extract principles from flow-matching literature; both use cross-scale consistency | RecFM reduces *truncation error* via consistency; QGF reduces *variance* via projection + Jacobian drop. Complementary — RecFM tightens the ODE, QGF guides it. |
| **204 NFCoT** (Normalizing Flow CoT) | Both score candidates by flow density | NFCoT scores *post-hoc*; QGF *steers generation*. QGF is the missing active counterpart to NFCoT's passive scoring. Plan 229 (NFCoT FlowScore) is MARGINAL — QGF unblocks it. |
| **215 ECHO** (Environment Prediction) | Both use a value/prediction signal to improve policy quality | ECHO trains on env tokens (model-based); QGF uses critic gradient at test time (modelless). ECHO's prediction quality correlation is the *justification* for QGF's critic guidance. |
| **217 TRD** (Trajectory Refined Distillation) | Both refine drafts using trajectory info | TRD refines via distillation loss (model-based); QGF refines via gradient guidance (modelless). Same target, different mechanism. |
| **229 ProgramAsWeights** (Spec-to-Compile) | Both compile external signal into inference-time steering | PAW compiles specs into constraints (symbolic); QGF compiles critic values into velocity bias (continuous). Different signal sources. |
| **267 Thicket** (Variance Probe Routing) | Both use variance for adaptive compute | Thicket routes *between methods* by variance; QGF adapts *guidance weight* by critic variance. QGF reuses Thicket's variance probe. |

**Key synergy:** QGF + NFCoT + ECHO + Thicket form a coherent test-time-compute stack:
```
ECHO (predict env) → Q-critic trained (riir-ai)
  ↓
QGF (gradient at projection) → steers generation
  ↓
NFCoT (flow density score) → ranks QGF-guided candidates
  ↓
Thicket (variance probe) → adapts guidance weight per-query
```

---

## 9. Self-Learning Adaptive CoT (per constraint 4)

QGF enables **self-learning adaptive CoT without LLM training**:

1. The Q-critic is trained offline (riir-ai, model-based) on game trajectories.
2. At inference time (katgpt-rs, modelless), the critic gradient steers the CoT.
3. As the BC reference policy is updated (via LoRA in riir-ai), the critic is retrained, and the guidance improves — **without any change to the modelless inference code**.

This is the "self-learning" loop: the critic learns offline, the inference adapts online. No LLM training at inference time. The adaptation is in the *guidance signal*, not the weights.

For pure-modelless self-learning (no critic retraining), the Q-gradient can be approximated by **rejection-sampled returns** (BFN-style): run N rollouts, compute empirical returns, use the return gradient as a proxy for the critic gradient. This is the modelless fallback when no trained critic is available (Freeze tier).

---

## 10. Tests / Examples — Before vs After (per constraint 6)

### Expected GOAT Criteria

| Metric | Baseline (no QGF) | Target (with QGF) | Measurement |
|---|---|---|---|
| First-attempt accuracy (Sudoku) | DDTree + NFCoT score | **+3-8%** | `sudoku_9x9` test suite |
| Speculative acceptance rate | DDTree baseline | **+5-12%** | speculative bench |
| Bomber win rate (vs heuristic) | Current best | **+2-5%** | arena integration |
| Guidance overhead | N/A | **< 2%** of inference | prof_bench |
| False-positive guidance (off-manifold) | N/A | **< 5%** | OOD detection test |

### Test Plan

1. **Unit:** First-order projection correctness (known prefix → known projection)
2. **Unit:** Q-gradient at projection matches finite-difference gradient (within tolerance)
3. **Unit:** Drop-Jacobian estimator has lower variance than BPTT estimator (cosine similarity test, per paper Fig 3)
4. **Integration:** QGuidedDrafter on Sudoku — before/after first-attempt accuracy
5. **Integration:** QGuidedDrafter on Rust syntax validation (SynPruner)
6. **GOAT:** Bomber arena — win rate with vs without QGF guidance
7. **Perf:** Overhead benchmark (< 2% target)

---

## 11. Open Questions

1. **Projection quality for discrete chains:** QGF's first-order projection works for continuous ODEs. For discrete token chains, what's the best projection — k-step greedy decode, or single drafter call with collapsed remaining budget? Need benchmark.
2. **Guidance weight schedule:** Fixed `1/β` per-domain (paper) or adaptive per-query (F4)? F4 is more general but riskier. Start with fixed, promote to adaptive after GOAT proof.
3. **Critic availability at inference:** If no trained critic is available (Freeze tier), fall back to rejection-sampled return gradient (BFN-style). What's the breakeven batch size where BFN-proxy beats no-guidance?
4. **Interaction with ThoughtFold:** QGF steers generation; ThoughtFold folds chains post-hoc. Can QGF guidance be applied *after* folding to re-steer the folded chain? Likely yes — fold then re-project then re-guide.

---

## 12. Implementation Plan Summary

See `katgpt-rs/.plans/268_qgf_test_time_q_guided_flow.md` for the full task breakdown.

**Phase 1 (unblock):** F2 FirstOrderProjector + F3 QGradientOracle trait — pure primitives, no integration risk.
**Phase 2 (core):** F1 QGuidedDrafter — integrate with SpeculativeGenerator + NFCoT FlowScore.
**Phase 3 (adaptive):** F4 VarianceAdaptiveGuidance — sigmoid-gated per-query weight.
**Phase 4 (routing):** CPU/SIMD/GPU/ANE auto-route + plasma/hot/warm/cold/freeze tiers.
**Phase 5 (GOAT):** Before/after benchmarks on Sudoku + Bomber + Rust syntax.

---

## References

1. Zhou, Peng, Xu, Li, Springenberg, Frans, Levine. "Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning." arXiv:2606.11087, Jun 2026. Code: github.com/zhouzypaul/qgf
2. Kostrikov, Nair, Levine. "Offline RL with Implicit Q-Learning." arXiv:2110.06169 (IQL — the critic training recipe, model-based, riir-ai).
3. Li, Zhou, Levine. "Reinforcement Learning with Action Chunking." arXiv:2507.07969 (BFN baseline).
4. Park, Li, Levine. "Flow Q-Learning." arXiv:2502.02538 (FQL — training-time distillation analogue).
5. Kang et al. "Efficient Diffusion Policies for Offline RL (EDP)." NeurIPS 2023 (first-order approximation inspiration).
6. Related research: 150 (RecFM), 204 (NFCoT), 215 (ECHO), 217 (TRD), 229 (PAW), 267 (Thicket).
