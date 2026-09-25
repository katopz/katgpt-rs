# Research 250: Latent Recursion = Policy Improvement — Self-Advantage from Pre/Post Logits

> **Source:** [Latent Reasoning in TRMs is Secretly a Policy Improvement Operator](https://arxiv.org/abs/2511.16886) — Asadulaev, Banerjee, Karray, Takac (MBZUAI), ICML 2026
> **Code:** [github.com/machinestein/Deep-Improvement-Supervision](https://github.com/machinestein/Deep-Improvement-Supervision) (PyTorch)
> **Date:** 2026-06-16
> **Status:** Active
> **Related Research:** 160 (SDPG — same math, bandit level), 049 (PTRM — parent TRM), 079 (EqR — recursion framing), 048 (HRM-Text — parent HRM)
> **Related Plans:** 180 (SDPG Bandit — shipped `centered_log_ratio`), 083 (PTRM width scaling), 119 (EqR convergence selector)
> **Classification:** Public

---

## TL;DR

The paper proves that a single latent-reasoning recursion step is a **policy improvement operator**: it produces two policies (reference `π̂` from pre-recursion logits, improved `π+` from post-recursion logits), and their **log-ratio `A(s,a) = log π+(a|s) − log π̂(a|s)` is an advantage-like signal** — no value function, no teacher, no oracle needed. This self-advantage answers "when does a recursion step help?" (Advantage Margin condition, Eq. 18): *iff the ground-truth action has above-average improvement score under the interpolated policy family*. The paper uses this to build the **DIS** training method (monotone discrete corruption targets → 18× fewer forward passes), but that training part belongs in `riir-train`.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **Self-Advantage** — the log-ratio between a model's own pre-recursion and post-recursion logits IS an advantage signal. No external teacher. Same math as SDPG's `centered_log_ratio` (Plan 180), but sourced from a single model's two passes instead of oracle-vs-student bandits.
2. **Advantage-Margin Gate** (Eq. 18) — accept a recursion step iff `A(s, y*) > E_{a∼π_w}[A(s,a)]`. Skip dead compute.
3. **Product-Policy Sharpening** (Eq. 12/16) — `π_w(a|s) ∝ π̂(a|s)^{1−w} · π+(a|s)^w`, a monotone multiplicative reweighting. Inference-time policy interpolation — we don't have this yet.

**Redirect → riir-train:** The DIS training method (discrete corruption schedule `y†_s`, per-step CE supervision) requires gradient updates and is out of scope here.

---

## 1. Paper Core Findings

### 1.1 The central theorem

For a TRM/HRM recursion step `t`, the state `s_t = (x, z^L_t, z^H_t)` produces **two output distributions at no extra cost**:

```
Reference policy (pre-reasoning):    π̂_t(a|s_t) = softmax(f_O(z^H_t))
Improved policy (post-reasoning):    π+_t(a|s_t) = softmax(f_O(z^H_{t+1}))
```

where `z^H_{t+1} = f^H_ϕ(z^H_t, z^L_{t+1})` and `z^L_{t+1} = f^L_ϕ(x, z^H_t, z^L_t)` is **conditioned on the input `x`**.

By Bayes' rule inversion, the post-reasoning policy can be read as an optimality-conditioned policy `π+_t ≈ p(a | s_t, o=1)`, yielding the **implicit optimality likelihood** (Eq. 14):

```
p(o=1 | s_t, a) ∝ π+_t(a|s_t) / π̂_t(a|s_t)
```

and therefore the **advantage-like improvement score** (Eq. 17):

```
A_t(s_t, a) := log π+_t(a|s_t) − log π̂_t(a|s_t) ≡ log p(o=1 | s_t, a) + const
```

### 1.2 The Advantage Margin condition (Eq. 18) — when does recursion help?

The paper proves (via the second derivative of the cross-entropy loss w.r.t. interpolation weight `w`) that **a latent-reasoning update improves prediction iff**:

```
A_t(s_t, y*) > E_{a∼π_{t,w}}[A_t(s_t, a)]
```

i.e., the reasoning step preferentially increases the relative log-probability of the *correct* action compared to typical alternatives. This is a **testable, per-step condition** — the first formal criterion for "is this recursion step dead compute?"

### 1.3 Product-policy improvement family (Eq. 12/16)

The KL-regularized policy improvement solution is a product policy:

```
π_w(a|s) ∝ π̂(a|s)^{1−w} · π+(a|s)^w     (Eq. 16)
```

for any `w ≥ 0`. This is the inference-time analog of exponentiated-advantage reweighting (Eq. 4) and recovers CFGRL (Frans et al., 2025) as the special case where `f(u) = exp(wu)`.

### 1.4 DIS training method (→ riir-train)

DIS supplies monotone intermediate targets `{y†_s}` via a discrete corruption schedule (token masking with decreasing rate `β_s`), and supervises the post-reasoning readout at each step: `L_DIS = Σ_s CE(ℓ^c_s, y†_s)`. Results: 18× fewer forward passes, 24% ARC-AGI-1 with 0.8M params, no halting head.

**This requires gradient updates → riir-train.** The theoretical insight above does not.

---

## 2. Distillation

### 2.1 The transferable primitive: Self-Advantage

The deepest transferable insight is **not** the corruption schedule (training) or the architecture (TRM). It is:

> **A single model, run twice (pre-recursion and post-recursion), produces a self-advantage signal via log-ratio. No teacher, no oracle, no value function.**

This is a modelless, inference-time, latent-to-latent operation. It works on any architecture with iterative refinement: HLA recurrent belief, looped transformers, DDTree depth, speculative draft/verify pairs, NPC thought cycles.

### 2.2 Three modelless primitives

| Primitive | Math | Operation | Existing analog? |
|-----------|------|-----------|------------------|
| **Self-Advantage** | `A(a) = log π+(a) − log π̂(a)` | Compute log-ratio between pre/post recursion logits | SDPG `centered_log_ratio` (same math, but oracle-vs-student, not self) |
| **Advantage-Margin Gate** | Accept step iff `A(y*) > E_{a∼π_w}[A(a)]` | Per-step dead-compute detector; skip if non-positive margin | `EarlyStopGate` (confidence threshold — different signal) |
| **Product-Policy Sharpening** | `π_w ∝ π̂^{1−w} · π+^w` | Multiplicative interpolation; `w=0` = no reasoning, `w=1` = full reasoning, `w>1` = extrapolation | **None** — we don't have inference-time policy interpolation |

### 2.3 Why this matters for our stack

Currently our recursion/loop infrastructure (`LoopMode::WeightShared`, `EarlyStopGate`, `BanditPruner::dual_cutoff`, `best_of_k_rollouts`) answers:
- *When to stop?* — confidence threshold (`EarlyStopGate`), bandit cutoff (`dual_cutoff`)
- *Which rollout to pick?* — bandit Q-values (`WidthSelectionMode`)

It does **not** answer:
- *Did this step improve anything?* — no per-step improvement signal
- *How much should I trust the reasoning?* — no controllable interpolation weight
- *Is this step dead compute?* — only detects low-confidence, not "no improvement"

The self-advantage fills all three gaps with a single computation.

### Fusion

**SDPG (Plan 160/180) × DIS-theory × EqR (Research 79):**

| Component | What it provides | From |
|-----------|-----------------|------|
| SDPG `centered_log_ratio` | The math — `A(a) = D̄ − log(p̄/q̄)` — **already shipped** at bandit-arm level | `katgpt-rs/crates/katgpt-pruners/src/sdpg/advantage.rs` |
| DIS theoretical lens | The insight that pre/post logits of a SINGLE model give the same advantage — no oracle needed | This paper |
| EqR recursion framework | The iterative reasoning substrate (`LoopMode`, `EarlyStopGate`, latent state evolution) | Research 079 |

**What the fusion produces that none alone can:**
SDPG currently needs oracle replay data (teacher Q-values from winning games). DIS-theory says: *throw away the oracle — your own model's pre/post logits ARE the teacher/student pair.* This means the dense per-step credit assignment that SDPG provides for bandits (Plan 180's main win) becomes available for **any iterative computation**, including HLA belief evolution, NPC thought cycles, and speculative draft/verify — with zero external data dependency.

**Cross-pollination candidates (not yet fused, tracking for future):**
- **HLA `evolve_hla`** (`katgpt-rs/crates/katgpt-sense/src/reconstruction.rs`) — the canonical "shipped without a research note" mechanism. Currently evolves latent state without a per-step improvement signal. Self-advantage could gate which HLA updates are worth keeping.
- **NPC curiosity** (riir-ai Research 041, 126, 127) — thousands of NPCs each "thinking" per tick. Most thoughts are dead compute. Self-advantage gate → skip non-improving thoughts → massive 20Hz tick budget savings.
- **Freeze/thaw** — snapshot the improvement direction vector `A(·)` per NPC personality. Versioned latent direction vectors (BLAKE3-committed).

---

## 3. Verdict

### **GOAT**

**One-line reasoning:** The policy-improvement theoretical lens lets us recycle SDPG's `centered_log_ratio` math as a **self-advantage** signal for latent recursion — provably detects dead compute steps (paper: 18× forward pass reduction) and enables a new inference-time product-policy sharpening operator, but early stopping as a *capability* already exists via `EarlyStopGate`.

### Novelty gate (Q1–Q4, honest assessment)

| Q | Criterion | Answer | Notes |
|---|-----------|--------|-------|
| Q1 | No prior art? | **NO** | Math primitive (`centered_log_ratio`) ships in SDPG (Plan 180). Recursion framework ships in EqR/PTRM/HLA. The *combination* (self-sourced log-ratio for recursion gating) is novel, but components exist. |
| Q2 | New class of behavior? | **Partial** | Dead-compute detection via advantage margin is new, but "early stopping" as capability exists (`EarlyStopGate`). Product-policy sharpening is genuinely new (no existing analog). |
| Q3 | Product selling point? | **Uncertain** | "NPCs never waste a thought cycle" is compelling *if* we ship latent-reasoning NPCs at MMORPG scale (itself unproven in production). For katgpt-rs engine: "18× fewer forward passes" is a measurable claim. |
| Q4 | Force multiplier? | **YES** | Connects SDPG (math), EqR (recursion), HLA (latent state), NPC curiosity (game AI), freeze/thaw (versioning). ≥2 pillars. |

**Since Q1 is NO (prior art exists for the math), Super-GOAT is not available.** → GOAT.

**Super-GOAT potential (deferred, not claimed):** If the MMORPG NPC application proves out — thousands of NPCs with per-tick advantage-margin gating saving measurable tick budget — the selling point solidifies. Re-evaluate after Plan implementation + game-side benchmark. Do **not** pre-claim Super-GOAT.

### Routing

| Artifact | Destination | Status |
|----------|------------|--------|
| Research note (this file) | `katgpt-rs/.research/250_*.md` | ✅ Created |
| Plan | `katgpt-rs/.plans/283_*.md` | Create next |
| Open primitive | `katgpt-rs/src/pruners/` or `crates/katgpt-core/src/` | Behind feature flag |
| DIS training method | **→ riir-train** | Redirect noted, no files created here |

---

## 4. Implementation Direction (preview — full plan in Plan 283)

### Phase 1: Self-Advantage computation (~150 LOC)
- `fn self_advantage(pre_logits: &[f32], post_logits: &[f32]) -> Vec<f32>` — returns `A(a) = log π+(a) − log π̂(a)` per action
- Reuse SDPG's `centered_log_ratio` internal math, but source both distributions from the same model's pre/post recursion
- Zero-allocation: pre-allocated scratch buffer, SIMD-friendly chunked loop

### Phase 2: Advantage-Margin Gate (~100 LOC)
- `AdvantageMarginGate` — wraps any `SpeculativeGenerator` or recursion loop
- Accept recursion step iff `A(y*_candidate) > E_{a}[A(a)]` (margin positive)
- Feature flag: `advantage_margin_gate` (opt-in, benchmark before default)

### Phase 3: Product-Policy Sharpening (~80 LOC)
- `product_policy(pre_logits, post_logits, w) -> Vec<f32>` — returns `π_w ∝ π̂^{1−w} · π+^w`
- Controllable `w`: 0.0 = skip reasoning, 1.0 = trust fully, >1.0 = extrapolate
- Feature flag: `product_policy_sharpen` (opt-in)

### GOAT gate
- Benchmark: recursion loops with vs without advantage-margin gate
- Metric: **forward passes saved** at matched output quality (paper claims 18×)
- Domain: HLA belief evolution on bomber arena, or DDTree speculative decode
- Promote to default if ≥2× forward pass reduction with no quality loss; demote `EarlyStopGate` if it loses

---

## 5. Cross-References

- `katgpt-rs/.research/160_SDPG_Self_Distilled_Policy_Gradient.md` — same math (`centered_log_ratio`), bandit level
- `katgpt-rs/.plans/180_sdpg_bandit_modelless.md` — shipped `SdpgBanditPruner` with `AdvantageMode::Sigmoid`
- `katgpt-rs/.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md` — parent TRM (architecture we improve on)
- `katgpt-rs/.research/079_EqR_Equilibrium_Reasoners.md` — recursion-as-attractor; uses fixed-point residual (different signal)
- `katgpt-rs/.research/048_HRM_Text_Hierarchical_Recurrent_Pretraining.md` — parent HRM
- `katgpt-rs/crates/katgpt-pruners/src/sdpg/advantage.rs` — shipped `centered_log_ratio` to reuse
- `katgpt-rs/src/pruners/bomber/sdpg_player.rs` — shipped `SdpgPlayer` with positive-advantage gating

## 6. References

- Asadulaev et al., "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator," ICML 2026. [arxiv:2511.16886](https://arxiv.org/abs/2511.16886)
- Frans et al., "Diffusion Guidance is a Controllable Policy Improvement Operator," 2025. [arxiv:2505.23458](https://arxiv.org/abs/2505.23458)
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks," 2025. [arxiv:2510.04871](https://arxiv.org/abs/2510.04871)
- Wang et al., "Hierarchical Reasoning Model," 2025. [arxiv:2506.21734](https://arxiv.org/abs/2506.21734)
- Sutton et al., "Reinforcement Learning: An Introduction," 1998. (policy improvement theorem)

> **Follow-up (2026-09-25):** the *stochastic* sibling of this paper — Baek et al., "Generative Recursive Reasoning" [arxiv:2605.19376](https://arxiv.org/abs/2605.19376) — distilled in [Research 590](590_GRAM_Generative_Recursive_Reasoning.md): guided (μ≠0) width rollouts, mass-conserving perturbation, decode-free latent selection → Issues 895 (katgpt-rs) / 1008 (riir-ai) + riir-train Plan 419.
