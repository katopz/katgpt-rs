# Research 179: RAGEN-2 — Template Collapse, SNR Filtering, and MI Proxies for Agent RL (Modelless Distillation)

> **Source:** RAGEN 2: Reward Variance-Driven Template Collapse in Multi-Turn Agent RL (arXiv:2604.06268)
> **Date:** 2026-06
> **Status:** GOAT Verdict — ✅ GAIN (Modelless Path: RV-Gated Compute Routing)

---

## TL;DR

RAGEN-2 identifies **template collapse** in multi-turn agent RL: reasoning becomes input-agnostic (same template across different inputs) while conditional entropy H(Z|X) remains high, making the collapse invisible to entropy-based monitoring. The paper introduces:

1. **MI proxy** I(X;Z) — in-batch cross-scoring of reasoning traces against prompts. Detects collapse that entropy alone misses.
2. **SNR mechanism** — low reward variance (RV) per prompt weakens task gradients (∥g_task∥ ≤ √RV × C by Cauchy-Schwarz), allowing regularization to dominate → input-agnostic updates → collapse.
3. **SNR-aware filtering** — keep top-ρ fraction of prompts by RV before each update. Top-ρ (nucleus-style) > Top-k > no filter. Reduces per-step time by 26–41%.

**Our verdict: GOAT for modelless.** RAGEN-2's core insight — *reward variance is an SNR proxy for useful learning signal* — maps directly to inference-time **acceptance variance across speculative decode attempts**. When acceptance variance is high, the model is uncertain → engage GPU + deeper DDTree + thinking mode. When low, CPU direct decode suffices. This is the missing signal for CPU/GPU auto-routing.

---

## Paper Core

### The Problem: Template Collapse

In multi-turn agent RL, reasoning tokens Z can collapse into input-agnostic templates while remaining high-entropy:

```
H(Z) = I(X;Z) + H(Z|X)

Template collapse:  I(X;Z) → 0  (reasoning decouples from input)
                  H(Z|X) stays high  (still varied, just not input-dependent)
```

Entropy monitoring H(Z) sees no anomaly because H(Z|X) dominates. The model "looks like it's thinking" but produces the same reasoning pattern regardless of the query.

### SNR Mechanism (Why It Happens)

Cauchy-Schwarz bound on per-prompt task gradient:

```
∥g_task(x)∥ ≤ √(Var[R(x)]) × C

When Var[R(x)] is low:
  - Task gradient is small
  - Regularization gradient (KL, entropy) is CONSTANT across prompts
  - Regularization dominates → input-agnostic updates → template
```

This is the fundamental mechanism: **low reward variance means low task SNR**, and regularization always has the same magnitude. When task signal is weak, the model converges to whatever regularization prefers — which is input-independent.

### MI Proxy (How to Detect It)

In-batch cross-scoring: for a batch of (prompt_i, reasoning_i) pairs, score reasoning_j against prompt_i for all i ≠ j. If the model can't distinguish its own reasoning from cross-paired reasoning → I(X;Z) ≈ 0 → collapse.

Two variants:
- **Retrieval-Acc** (discrete): rank reasonings by relevance to each prompt, check if correct pairing is retrieved
- **MI-ZScore-EMA** (continuous): z-scored log-prob differences with exponential moving average

### Key Numbers

| Metric | No Filter | Top-ρ Filter | Δ |
|--------|-----------|--------------|---|
| MI proxy (higher = better) | 0.23 | 0.71 | +209% |
| Task score | 3.4 | 5.8 | +71% |
| Per-step time | 1.0× | 0.59–0.74× | 26–41% faster |
| Filter compute overhead | — | <0.1% | Negligible |

---

## Modelless Distillation: RV as Inference-Time Signal

### The Core Mapping

RAGEN-2's insight is **domain-agnostic**: when the variance of a signal is low, the signal carries little information. This applies beyond RL training.

```
Training domain:    Var[R(x)] per prompt → SNR of task gradient → template collapse
Inference domain:   Var[acceptance] per query → SNR of draft quality → compute allocation
```

### What We Extract (No Training Required)

The paper's contribution for us is not the MI proxy or the filtering algorithm — it's the **principle** that variance of a per-sample signal is an SNR proxy, and that filtering low-SNR samples before acting improves outcomes at lower cost.

This principle maps to three concrete inference-time mechanisms:

### Fusion A: RV-Gated Compute Routing (GOAT — Default ON if Gain Proven)

**The insight:** At inference time, we don't have "reward variance" from RL, but we have **acceptance variance across speculative decode attempts**.

```
Training:   Var[episode returns per prompt]     = RV
Inference:  Var[acceptance rates per query]     = RV_analog
```

When a query has high acceptance variance → model is uncertain → promote compute tier (GPU + deeper DDTree + Latent thinking).

When acceptance variance is low → model is confident or templating → CPU direct decode is fine.

**Implementation map:**

```rust
/// Tracks per-query acceptance variance from speculative decode stats.
/// This is the modelless analog of RAGEN-2's reward variance (RV).
///
/// RAGEN-2 proves: low RV = low task SNR = regularization dominates = template collapse.
/// Our analog: low acceptance variance = model is confident = cheap path OK.
///             high acceptance variance = model is uncertain = expensive path needed.
pub struct AcceptanceVarianceTracker {
    /// Welford's online variance — we already have this in BanditStats
    mean: f64,
    m2: f64,
    count: u64,
    /// EMA-smoothed RV for routing decisions (RAGEN-2 uses EMA too)
    ema_rv: f64,
    ema_alpha: f64,
}

impl AcceptanceVarianceTracker {
    /// Update with acceptance outcome from one speculative decode attempt.
    pub fn observe(&mut self, accepted: bool) { ... }

    /// Current RV estimate. Feeds into InferenceRouter + ThinkingController.
    pub fn rv(&self) -> f64 { ... }
}
```

**Routing integration:**

| RV Level | InferenceRouter Tier | ThinkingController Mode | DDTree Depth |
|----------|---------------------|------------------------|--------------|
| High (σ² > θ_high) | GPU | Latent | Deep (full) |
| Medium (θ_low < σ² ≤ θ_high) | GPU/CPU | Latent/Direct | Medium |
| Low (σ² ≤ θ_low) | CPU | Direct | Shallow |

**Why this works with existing infrastructure:**

| Component | Already Has | What We Add |
|-----------|-------------|-------------|
| `BanditStats` | `reward_variance()` via Welford | `AcceptanceVarianceTracker` reuses same math |
| `InferenceRouter` | QPS/load-based routing | RV as additional routing signal |
| `TriggerGate` | CPU/GPU/ANE tier gates | RV-gated promotion/demotion |
| `ThinkingController` (Plan 194) | QPS-gated thinking mode | RV-gated thinking mode |
| `FrequencyBandit` | UCB1 arm selection | RV as arm context feature |

### Fusion B: MI-Gated Draft Acceptance (Secondary)

**The insight:** RAGEN-2's MI proxy detects when reasoning is input-agnostic. At inference time, detect when the draft model's continuations are input-dependent.

```
Cross-score recent N draft sequences against cached contexts.
If retrieval accuracy < chance → draft is templating → re-temperature or re-draft.
```

This slots into `ScreeningPruner` as an additional relevance signal:

```rust
/// MI-proxy draft scorer. Detects input-agnostic (template) drafts.
///
/// RAGEN-2: I(X;Z) = Σ_x Σ_z p(x,z) log(p(x,z) / (p(x)·p(z)))
/// Modelless: Approximate via retrieval accuracy of drafts against context cache.
pub trait MiProxyScorer: ScreeningPruner {
    /// Score draft's input-dependence via cross-retrieval against cached contexts.
    /// Returns f32 in [0,1]: 1 = fully input-dependent (good), 0 = templating (bad).
    fn mi_score(&self, draft: &[TokenId], context_cache: &ContextCache) -> f32;
}
```

**Trade-off:** Requires maintaining a context cache (O(batch_size × context_len)). Only viable when batch inference is already happening. Lower priority than Fusion A.

### Fusion C: SNR-Bandit Arm Pruning (Incremental)

**The insight:** RAGEN-2's quartile ablation shows training only on high-RV prompts is best. At inference, prune bandit arms with low reward variance (no learning signal).

**Already partially implemented:** `BanditStats::reward_variance()` computes Welford variance per arm. Extend to actively suppress low-RV arms:

```rust
/// During prepare_episode(), suppress arms with RV below threshold.
/// RAGEN-2 shows: top-ρ (nucleus) filtering > top-k > no filter.
fn prune_low_rv_arms(bandit: &mut FrequencyBandit, rho: f32) {
    let arms = bandit.arm_variances();
    let threshold = arms.quantile(1.0 - rho); // top-ρ nucleus style
    bandit.suppress_arms_below(threshold);
}
```

**Effort:** Minimal — `BanditStats` already has the variance. Just add the suppression gate.

---

## GOAT Verdict

### Per 003_Commercial_Open_Source_Strategy_Verdict.md

| Criterion | Verdict | Reasoning |
|-----------|---------|-----------|
| **Modelless first** | ✅ | All three fusions are inference-time only. No LLM training. |
| **Engine/Fuel split** | ✅ | RV-gated routing goes into katgpt-rs (MIT engine). Accumulated RV data is deployment-specific fuel. |
| **No perf hurt** | ✅ | `AcceptanceVarianceTracker` is O(1) per observe(). Welford update is 3 flops. Routing check is one comparison. |
| **SOLID/DRY** | ✅ | Reuses `BanditStats` Welford math. New tracker is a separate struct composed into `InferenceRouter`. |
| **Default ON if GOAT+gain** | ✅ | Feature-gated as `rv_gated_routing`, default on after benchmark proves no regression. |
| **Tests/examples** | ✅ | Before/after: latency vs quality tradeoff on synthetic acceptance-variance distributions. |

### Why Fusion A Is GOAT

1. **Novel domain mapping** — RAGEN-2's RV insight is about RL training. We map it to inference compute allocation. This is not a direct port; it's a creative transposition.
2. **Solves a real requirement** — "CPU/GPU auto-route when load changes" is an existing requirement. RV is the missing signal beyond QPS.
3. **Uses existing infrastructure** — `BanditStats`, `InferenceRouter`, `TriggerGate`, `ThinkingController`, `FrequencyBandit` all already exist. The fusion is additive.
4. **Zero-cost when wrong** — If RV signal is noisy, the router falls back to QPS-based routing. No harm.
5. **Measurable** — Before/after: latency quantiles at same quality, or quality at same latency budget.

### What We DON'T Take

| Paper Concept | Why Rejected |
|---------------|-------------|
| **MI proxy training loss** | Requires LLM training. Goes to riir-ai (model-based path), not katgpt-rs. |
| **GRPO + RV filtering** | Training-time mechanism. Not applicable to inference engine. |
| **Retrieval-Acc metric** | Requires batched cross-scoring of reasoning traces. Not feasible at inference latency budgets. |
| **MI-ZScore-EMA** | Continuous variant of retrieval metric. Same latency concern. |
| **Top-ρ prompt filtering** | Training-time data culling. We adapt the *principle* (variance-gated compute) not the mechanism. |
| **Multi-turn episode tracking** | Our queries are single-turn inference. Episode-level RV doesn't apply directly. |

---

## Actionable Mapping Table

| Paper Concept | katgpt-rs Analog | Component | Plan | Priority |
|---------------|-----------------|-----------|------|----------|
| Reward variance (RV) | Acceptance variance per query | `AcceptanceVarianceTracker` | New | 🔴 GOAT |
| SNR-aware filtering | RV-gated compute tier routing | `InferenceRouter` + `TriggerGate` | Extend | 🔴 GOAT |
| Top-ρ nucleus filtering | Top-ρ bandit arm suppression | `FrequencyBandit` | Extend | 🟡 High |
| MI proxy I(X;Z) | Cross-retrieval draft scoring | `ScreeningPruner` | New | 🟢 Secondary |
| EMA smoothing | EMA on RV signal | `AcceptanceVarianceTracker` | New | 🔴 GOAT |
| Template collapse detection | Acceptance variance threshold | `ThinkingController` (Plan 194) | Extend | 🟡 High |

### Implementation Order

```
1. AcceptanceVarianceTracker (Welford + EMA)           — ~50 lines, standalone
2. Wire RV into InferenceRouter as routing signal        — ~30 lines, additive
3. Wire RV into ThinkingController mode selection        — ~20 lines, additive
4. Top-ρ bandit arm suppression in FrequencyBandit       — ~20 lines, extends BanditStats
5. MiProxyScorer for ScreeningPruner (if batch avail)    — ~80 lines, secondary
```

### GOAT Gate Feature Flag

```toml
[katgpt-rs.features]
rv_gated_routing = true      # GOAT — default ON after benchmark
rv_gated_thinking = true     # GOAT — default ON after benchmark
rv_bandit_pruning = false    # Needs ablation first
mi_draft_scoring = false     # Secondary, needs context cache
```

### Benchmark Proving Path

```
1. Generate synthetic acceptance-variance distributions (bimodal: confident + uncertain queries)
2. Measure P50/P99 latency with RV-gated routing ON vs OFF
3. Measure quality (acceptance rate, perplexity) at same latency budget
4. Prove: RV-gated routing ≥ same quality at ≤ same P99, with improved P50 for confident queries
5. If proven → default ON. If not → feature flag stays off, no perf hurt.
```

---

## Negative Results / What We Reject

1. **Direct MI proxy at inference**: Cross-scoring reasoning traces against prompts requires batched inference + scoring model. Latency budget at inference is ms-scale. The training-time luxury of "score every batch combination" doesn't exist here. We adapt the *principle* (input-dependence check) but not the mechanism.

2. **Entropy monitoring**: RAGEN-2's whole point is that entropy H(Z) is insufficient — H(Z|X) can be high during collapse. We should NOT add entropy-based monitoring to our inference pipeline. RV is the correct signal.

3. **Top-k filtering over Top-ρ**: The paper shows nucleus-style (Top-ρ by variance) beats Top-k. If we implement bandit arm pruning, use quantile thresholding not fixed-k.

4. **Per-token RV**: RAGEN-2 operates at prompt level, not token level. Per-token acceptance variance is too noisy (binary per position). Aggregate per-query is the right granularity.

---

## Related Research in katgpt-rs

| Doc | Relation |
|-----|----------|
| 075 Survive or Collapse | Data gating in self-play — complementary filtering philosophy |
| 078 MTP Cluster Top-K | BanditStats has `reward_variance()` — foundation for RV tracking |
| 098 PrudentBanker | Safe bandit arm selection — RV pruning extends this |
| 155 ANE Compute Backend | Target backend for low-RV (confident) routing |
| 156 Speculative Reconciliation | Acceptance rates are the RV source signal |
| 162 Trust Region Adaptive Speculation | Adaptive speculation parameters — RV can inform trust region |
| 169 Oscillatory State Space | Spectral analysis — RV variance is a spectral feature |
| 177 Domino | Speculative decode architecture — acceptance variance comes from here |
| Plan 194 ThinkingController | Adaptive CoT — RV feeds into thinking mode selection |

---

## TL;DR

RAGEN-2 proves that **reward variance is an SNR proxy**: low variance = weak task gradient = regularization dominates = template collapse. We map this to inference: **acceptance variance across speculative decode attempts is an SNR proxy for compute need**. High acceptance variance → uncertain query → promote to GPU + Latent thinking. Low acceptance variance → confident query → CPU direct decode. This is Fusion A (RV-Gated Compute Routing), our GOAT adoption. It's modelless, additive, uses existing infrastructure (`BanditStats`, `InferenceRouter`, `TriggerGate`, `ThinkingController`), and defaults to current behavior when disabled. Feature-gate as `rv_gated_routing`, benchmark, default ON if gain proven.
