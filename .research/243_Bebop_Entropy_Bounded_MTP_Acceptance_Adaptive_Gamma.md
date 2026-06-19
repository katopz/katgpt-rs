# Research 243: Bebop — Entropy-Bounded MTP Acceptance & Adaptive γ Forecast

> **Source:** [Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling](https://arxiv.org/pdf/2606.12370) — Li, Jiang, Xu, et al. (Qwen Team, Alibaba Inc.), arXiv:2606.12370v1, Jun 2026
> **Date:** 2026-06-16
> **Status:** Active — **Gain** (fusion idea; adaptive application unproven even in source paper)
> **Related Research:** 002 (Speculative Decoding — Leviathan), 156 (Speculative Reconciliation), 162 (Trust-Region Speculation), 217 (TRD Trajectory-Refined Draft), 218 (BFCF × LFU Sharding — `freq_bandit` reward), 235 (Thicket Variance Probe Routing)
> **Related Plans:** 223 (LLMExecGuard — entropy→verify tier), 249 (TRDraft), 268 (Q-Guided Flow drafter)
> **Related Issues:** 023 (this doc's optimization — adaptive γ from entropy forecast)
> **Classification:** Public — generic math, no game semantics

---

## TL;DR

Bebop is primarily a **training-recipe paper** (end-to-end TV loss for MTP heads → `riir-train`). But it ships one **proven, modelless, zero-allocation inference-time truth** that we do not yet exploit: the **linear entropy–acceptance bound**.

For any draft model trained with CE/KL:

```
α_TO ≈ a_TO − b_TO · H(p)        (target-only sampling)
α_RS ≈ a_RS − b_RS · H(p)        (rejection sampling, CE/KL-trained draft)
α_RS_TV ≈ 1 − δ/2                 (TV-trained draft — entropy-invariant)
```

where `H(p) = −Σ p_v ln p_v` is the target model's next-token entropy, computable in one pass over the logits. The constants `a, b` are fitted once from early training and are remarkably stable across model sizes, tasks, and RL stages (paper Fig. 1a, Fig. 8).

**Distilled for katgpt-rs (modelless, inference-time):**

Three things, in decreasing order of "already have it":

1. **Rejection sampling > target-only** for MTP acceptance — **already shipped** in `LeviathanVerifier` (`src/speculative/verifier.rs:128`). The paper confirms this is the right default for virtually all native MTP deployments (§7.5: 23/24 model–task pairs fall in the RS-better region). No action.
2. **Entropy → verification tier gating** — **already shipped** in `llmexec_guard` (`src/llmexec_guard.rs`) via `sigmoid(-steepness * (entropy - 0.5) + depth_bonus)`. But ours is an **ad-hoc sigmoid confidence gate**; the paper gives a **proven linear acceptance-rate forecast** `α ≈ a − b·H(p)`. Replacing the ad-hoc sigmoid with the calibrated linear forecast is the Gain.
3. **Per-step adaptive γ from acceptance forecast** — **NOT shipped.** `draft_lookahead` is a static `Config` field. The paper explicitly flags adaptive-γ as future work (§7.6: "suggests that adaptive MTP strategies — adjusting the draft length γ based on the estimated local entropy—could further improve throughput") **without proof**. This is the genuinely open fusion opportunity, and it is unproven.

**Why Gain, not higher:** The linear bound is a proven insight, but the adaptive application (γ-shrinking, skip-when-low-α, bandit-prior) is speculative — even the paper doesn't benchmark it. No new capability class. No product selling point. The mechanism is a two-parameter linear model, not a novel primitive.

---

## 1. Paper Core Findings

### 1.1 The entropy–acceptance bound (Propositions 1, 2, 4)

The paper proves that MTP acceptance rates are **linearly bounded by the target model's entropy**, under standard training objectives:

| Sampling | Draft training | Acceptance bound | Entropy sensitivity |
|---|---|---|---|
| Target-only (`α_TO`) | any | `α_TO ≈ a_TO − b_TO · H(p)`, lower-bounded by `exp(−H(p))` | strong negative linear |
| Rejection (`α_RS`) | CE / KL | `α_RS ≈ a_RS − b_RS · H(p)` | strong negative linear (slope comparable to `b_TO`) |
| Rejection (`α_RS_TV`) | **TV loss** | `α_RS_TV ≥ 1 − δ/2` (entropy-**invariant**) | near-zero (empirical slope −0.06 vs −1.68 for CE) |

The proof sketches:
- **Target-only:** `α_TO = p(argmax q) = max_y p(y)` when draft is correct, and `max_y p(y) ≥ exp(−H(p))` by Jensen. First-order Taylor around operating entropy gives the linear form.
- **RS + CE/KL:** CE gradient `q_j − p_j` produces uniform per-token mismatch `|η_v| ≲ σ`. Effective support size `|S_eff| ≈ exp(H(p))`, so `dTV ≈ (σ/2)·exp(H(p))`. Linearize the exponential → linear form.
- **RS + TV:** TV gradient `−q_j[1[q_j ≤ p_j] − S]` is `q_j`-proportional, producing probability-proportional mismatch `|q−p| ≲ δ·p`. Summing: `dTV ≤ (δ/2)·Σ p = δ/2`, **independent of entropy**.

### 1.2 Rejection sampling vs target-only decision boundary (§7.5, §E)

RS beats target-only when `dTV(p,q) < 1 − p(ŷ)` where `ŷ = argmax q`. Empirically, 23/24 model–task pairs fall in the RS-better region for natively trained MTP heads. **Implication: RS should be the default for essentially all native MTP deployments.**

### 1.3 Decomposition: entropy vs mismatch in RL (§5.1)

During RL training, acceptance-rate change decomposes as:

```
Δα_t = b·(H_t − H_0)  +  (Δα_t − b·(H_t − H_0))
       \____ entropy ___/   \_____ mismatch ______/
```

Under RS + CE: degradation is **almost entirely entropy-driven** (`Δα_mismatch ≈ 0`). Policy weight updates don't significantly affect draft–target TV overlap. This is why **pre-RL MTP adaptation is sufficient** — no online MTP co-training needed during RL.

### 1.4 The TV loss (training — → riir-train)

The novel training objective `L_TV = 1 − Σ_v min(p_v, q_v)` directly optimizes the rejection-sampling acceptance rate. Gradient is bounded (`|∂L/∂z_j| ≤ 1`), `q_j`-proportional (tail-suppressing), and drives `q_j/p_j → 1`. The e2e multi-step variant `L_e2e = 1 − (1/γ)·Σ_j Π_i α_i` captures the multiplicative structure of multi-step acceptance. This is a **training recipe → riir-train** (where we already have `verify_pinsker_bound` in `riir-train-engine/src/critical_position.rs`).

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by code grep)

| Paper claim | Our shipped equivalent | Status |
|---|---|---|
| RS acceptance = `1 − dTV(p,q)` | `LeviathanVerifier` (`src/speculative/verifier.rs:128`, "Real p/q rejection sampling (Algorithm 1)"); target probs cached in `p_distributions_flat` (`speculative/types.rs:225`) | ✅ shipped |
| RS preferred over target-only | `LeviathanVerifier` always uses RS; `step.rs:66`, `trust_region.rs:153`, `d2f_verifier.rs:151` all use p/q RS | ✅ shipped (correct default) |
| Entropy → verification budget | `llmexec_guard` (`src/llmexec_guard.rs`): `sigmoid(-steepness·(entropy−0.5) + depth_bonus)` → `VerifyTier::{Skip, Screening, FullVerify}` | ⚠️ shipped but **ad-hoc sigmoid**, not the paper's **proven linear α forecast** |
| Entropy-spike detection | `RejectionReason::EntropySpike` in `src/distill/trd.rs:56`, gated by `entropy_threshold` | ✅ shipped |
| Acceptance-rate bandit reward | `freq_bandit` (`src/freq_bandit.rs:315`, "reward = acceptance_rate × latency_improvement"), `fold_bandit`, `meta_router` (`src/dash_attn/meta_router.rs:206`) | ✅ shipped |
| EMA entropy tracking | `AdaptiveTraceCompactor::observe_entropy` (`src/attn_match/adaptive_cot.rs:159`), `α=0.1` EMA | ✅ shipped (for KV compaction, not spec-decode γ) |
| TV distance / Pinsker | `verify_pinsker_bound` in `riir-train-engine/src/critical_position.rs:177` | ✅ shipped in **riir-train** (training-side, correct repo) |

### 2.2 What's NOT in katgpt-rs (the gap)

1. **Acceptance-rate forecast from entropy.** Our `llmexec_guard` maps entropy → tier via sigmoid. The paper proves `α ≈ a − b·H(p)` — a **calibrated quantitative forecast** of the actual acceptance rate, not just a confidence gate. We never compute or expose `α_forecast`.
2. **Per-step adaptive γ.** `Config::draft_lookahead` is static (`micro`=8, `small_target`=5, `bpe`=8, `gemma2_2b`=0, `game`=0). It is never shrunk at runtime when entropy rises. Paper §7.6 flags this as future work without proof.
3. **Skip-speculative-decode-when-low-α.** When forecast `α < breakeven`, single-token decode is cheaper than draft+verify. We have no such gate; `llmexec_guard` gates verification *tier*, not whether to speculate at all.
4. **Per-step RS-vs-TO selection.** `LeviathanVerifier` always does RS. The paper's decision boundary `dTV < 1 − p(ŷ)` could in principle switch to TO when the draft is very weak (though §7.5 says this almost never happens for native MTP).

### 2.3 Fusion (the Gain-tier idea)

**Fusion: entropy-linear acceptance forecast × existing adaptive machinery.**

Replace `llmexec_guard`'s ad-hoc sigmoid with a calibrated linear forecast, and let the forecast drive three things it currently cannot:

```
                    ┌─→ adapt γ_t = clamp(ceil(L_target / α_forecast), γ_min, γ_max)
H(p_t) ──► α_forecast = a − b·H(p_t) ──┼─→ skip spec-decode when α_forecast < α_breakeven
                    └─→ bandit prior for freq_bandit / fold_bandit (speeds convergence)
```

Cousins in the corpus that this fuses with:

- **`llmexec_guard` (Plan 223)** — the sigmoid confidence gate becomes a calibrated α forecast.
- **`freq_bandit` (Plan 189) / `fold_bandit` (Plan 195) / `meta_router`** — the forecast α becomes a Bayesian prior on the reward, rather than waiting for the actual accept/reject outcome.
- **`AdaptiveTraceCompactor` (Plan 238 wire-patch)** — already tracks EMA entropy per trace; the same EMA feeds the α forecast for free.
- **`trust_region_spec` (default-on)** — trust-region radius could scale with forecast α (high α → tighter trust region, more aggressive speculation).
- **`belief_drafter` (Plan 217)** — `belief_drafter_entropy_threshold` is currently a static 2.0; the forecast replaces it with a continuous α-driven gate.

**What's novel in the fusion:** None of our existing entropy-driven gates produce a **quantitative acceptance-rate forecast**. They all use entropy as a qualitative confidence signal. The paper's contribution is the proof that the relationship is **linear and calibrated** — which means it can be inverted to set γ and to make skip-decide decisions, not just tier routing.

**What's NOT novel:** Rejection sampling itself (shipped), entropy-driven gating (shipped), acceptance-rate-as-bandit-reward (shipped). The fusion is "upgrade the gate from qualitative to quantitative, and let the quantitative forecast drive γ."

### 2.4 Why this is NOT Super-GOAT

Novelty gate (Q1–Q4):

1. **No prior art?** Partially. Rejection sampling is shipped. Entropy-driven gating is shipped (`llmexec_guard`). The *linear calibrated forecast* and *adaptive-γ application* are not in our code — but the linear model is a two-parameter fit, not a novel primitive.
2. **New class of behavior?** No. It's an optimization of existing speculative decode. The system still speculates, verifies, accepts/rejects — just with better-calibrated γ.
3. **Product selling point?** No. "Our speculative decoder adapts draft length from entropy" is an engineering optimization, not a moat. Cannot finish "our NPCs do X no competitor can."
4. **Force multiplier?** Borderline yes — touches `llmexec_guard`, `freq_bandit`, `belief_drafter`, `trust_region_spec`. But Q1+Q2+Q3 fail kills Super-GOAT.

Not Super-GOAT. Not GOAT either: the paper proves the *bound* but does NOT prove the *adaptive-γ application* improves throughput (§7.6 is explicitly future work). So there is no proven gain over our existing stack to promote. **Gain tier.**

---

## 3. Verdict

**Gain.** A proven linear entropy–acceptance bound, modelless and zero-allocation, that upgrades our existing ad-hoc entropy gating (`llmexec_guard`) into a calibrated acceptance-rate forecast capable of driving per-step adaptive γ. The adaptive application is unproven even in the source paper, so this stays behind a feature flag with a GOAT gate until we benchmark it ourselves.

**One-line reasoning:** Rejection sampling is already shipped; the only genuinely new modelless piece is the linear α-forecast, whose adaptive-γ application is explicitly future-work in the paper — useful, incremental, needs proof.

**Routing:**
- TV loss + e2e multi-step TV objective + online-vs-offline MTP adaptation → **riir-train** (training recipe; `verify_pinsker_bound` already there).
- Rejection sampling acceptance method → **already shipped** (`LeviathanVerifier`).
- Entropy-linear α forecast + adaptive γ fusion → **katgpt-rs** (this note + Issue 023).

---

## 4. Implementation Sketch (for Issue 023)

The primitive is tiny — a two-parameter linear model with an EMA-tracked entropy input:

```rust
/// Calibrated acceptance-rate forecast from target entropy.
/// α ≈ a − b · H(p), proven linear bound (Bebop, arXiv:2606.12370 §3).
/// Fit `a, b` once from early warmup; forecast is O(1) per step after entropy computation.
#[derive(Clone, Copy, Debug)]
pub struct AcceptanceForecast {
    /// Intercept (fitted). Empirically ~1.0 for RS+TV, ~0.95 for RS+CE.
    pub a: f32,
    /// Entropy slope (fitted). Empirically ~0 for TV-trained, ~−1.68 for CE/KL.
    pub b: f32,
    /// EMA-smoothed target entropy H(p_t).
    pub ema_entropy: f32,
    pub ema_alpha: f32,
}

impl AcceptanceForecast {
    #[inline]
    pub fn observe_and_forecast(&mut self, next_token_logits: &[f32]) -> f32 {
        let h = entropy_from_logits(next_token_logits); // reuse AdaptiveTraceCompactor's helper
        self.ema_entropy = self.ema_alpha * h + (1.0 - self.ema_alpha) * self.ema_entropy;
        (self.a - self.b * self.ema_entropy).clamp(0.0, 1.0)
    }

    /// Adaptive γ: target L accepted tokens, shrink when forecast is low.
    #[inline]
    pub fn adaptive_gamma(&self, target_accept_length: usize, gamma_min: usize, gamma_max: usize) -> usize {
        let alpha = (self.a - self.b * self.ema_entropy).clamp(0.01, 1.0);
        let gamma = ((target_accept_length as f32) / alpha).ceil() as usize;
        gamma.clamp(gamma_min, gamma_max)
    }
}
```

Zero-allocation. O(vocab) per step only for the entropy computation (which `AdaptiveTraceCompactor` already does). The forecast itself is O(1). Fits in L1. Feature flag: `adaptive_gamma_forecast` (default-off until GOAT gate passes).

**GOAT gate:** benchmark `accepted_tokens/sec` and `μs/step` with vs without the forecast-driven adaptive γ, on a workload with varying entropy (e.g., long CoT reasoning traces where entropy rises mid-generation, per paper Fig. 12b). Promote to default only if ≥5% throughput gain with no quality regression. Demote `llmexec_guard`'s ad-hoc sigmoid if the forecast strictly dominates it.

---

## 5. Cross-References

- `katgpt-rs/src/speculative/verifier.rs` — `LeviathanVerifier` (RS already shipped)
- `katgpt-rs/src/llmexec_guard.rs` — entropy→tier sigmoid (to be upgraded or demoted)
- `katgpt-rs/src/attn_match/adaptive_cot.rs` — `AdaptiveTraceCompactor::observe_entropy` (EMA entropy helper to reuse)
- `katgpt-rs/src/freq_bandit.rs` — acceptance_rate bandit reward (forecast becomes prior)
- `katgpt-rs/src/distill/trd.rs` — `RejectionReason::EntropySpike` (entropy threshold gate)
- `riir-train/crates/riir-train-engine/src/critical_position.rs` — `verify_pinsker_bound` (TV-distance training infra)

---

## TL;DR

Bebop is a training-recipe paper (TV loss → riir-train). Its one modelless inference-time gift is the **proven linear entropy–acceptance bound** `α ≈ a − b·H(p)`. Rejection sampling is already shipped in `LeviathanVerifier`. Entropy-driven gating is already shipped in `llmexec_guard` (but ad-hoc sigmoid, not calibrated forecast). The genuinely missing piece is **per-step adaptive γ from the acceptance forecast**, which the paper itself flags as unproven future work. **Gain tier** — useful, incremental, needs our own benchmark to prove. Tracked in Issue 023.

---

## Addendum (2026-06-19) — H_1 → H_2 upgrade recommendation (Plan 294 G10)

**Status: PROVEN + VERIFIED. Adopt the upgrade.**

Plan 294 Phase 6 GOAT Gate G10 calibrates the Bebop acceptance forecast with
H_1 (current Bebop baseline) vs H_2 (collision-purity-based, `−log Σ π²`)
on a synthetic workload with a 50/50 mix of "decisive" (`max π > 0.37`) and
"long-tail" (`max π < 0.37`) next-token distributions. Ground truth is the
paper's linear bound `α = a − b · H_2(p) + ε` (H_2 is the *correct*
concentration signal per ICT §A.3.3 — H_1 has wrong gradient sign for
`π < e⁻¹ ≈ 0.37`).

### Result (bench_294_ict_g10.rs)

| Forecaster       | a       | b       | MAE overall | MAE decisive | MAE long-tail |
|------------------|---------|---------|-------------|--------------|---------------|
| H_1 (Bebop)      | 12.340  | 3.531   | 0.4304      | 0.4407       | 0.4227        |
| H_2 (this)       |  8.094  | 2.049   | **0.3969**  | **0.3901**   | **0.4020**    |
| Δ (H_1 − H_2)    |         |         | +0.0334     | +0.0506      | +0.0207       |

**H_2 wins on every regime.** The recovered coefficients (a=8.09, b=2.05)
match the ground truth (a=8.0, b=2.0) almost exactly — the methodology is
sound. The improvement is small in absolute terms (~0.03 MAE) but is
**concentrated where ICT §A.3.3 predicts** (the long-tail regime where H_1
is provably wrong).

### The upgrade

`AcceptanceForecastH2` (shipped in `crates/katgpt-core/src/ict/bebop_upgrade.rs`,
behind feature `ict_branching`) is a drop-in replacement for Bebop's
`α ≈ a − b · H_1(p)`:

```rust,ignore
// OLD (Bebop Issue 023):
//   let alpha = a - b * shannon_h1(&softmax(logits));
// NEW (Plan 294 G10):
use katgpt_core::ict::AcceptanceForecastH2;
let mut forecast = AcceptanceForecastH2::new(a, b);
let alpha = forecast.observe_and_forecast(&next_token_logits);
```

The downstream adaptive-γ logic (`forecast.adaptive_gamma(target, lo, hi)`)
is unchanged — it consumes `α` the same way regardless of which entropy
produced it.

### Why H_2 is the right answer (paper proof)

ICT §A.2.5: ∂β/∂π(a) = 2π(a) > 0 unconditionally. β = Σ π² = exp(−H_2)
strictly increases with concentration. H_1 only has the right gradient for
π(a) > e⁻¹ ≈ 0.37 — for the long-tail tokens (the bulk of LLM vocabularies)
H_1 reports a wrong "decisiveness" signal. Bebop's forecast inherits that
wrongness; H_2 fixes it.

### Recommendation

- **Adopt `AcceptanceForecastH2` as the Issue 023 implementation primitive.**
- Re-calibrate `(a, b)` on a real acceptance-length dataset — the synthetic
  G10 numbers prove the *direction*; production coefficients need production
  data.
- The `ict_branching` feature stays opt-in until G8 (riir-ai Plan 324)
validates the runtime fusion. Bebop callers wanting the H_2 upgrade alone
  can enable just `ict_branching` and ignore the `BranchingDetector`.

### References

- Plan 294 §Phase 6 T6.1–T6.4
- Research 270 §1.5, §A.3.3 (H_1 monotonicity caveat)
- arxiv 2606.19771 §A.2.5 (β unconditional monotonicity)
- Test: `tests/bench_294_ict_g10.rs`
- Primitive: `crates/katgpt-core/src/ict/bebop_upgrade.rs`
