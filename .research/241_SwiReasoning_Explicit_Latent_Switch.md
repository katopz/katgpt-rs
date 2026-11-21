# Research 241: SwiReasoning — Switch-Thinking in Latent and Explicit

> **Source:** [SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs](https://arxiv.org/pdf/2510.05069) — Shi, Asi, Li, Yuan, Pan, Lee, Xiao (Georgia Tech / Microsoft), ICLR 2026
> **Code:** [github.com/sdc17/SwiReasoning](https://github.com/sdc17/SwiReasoning)
> **Date:** 2026-06-15
> **Status:** Active — GOAT distillation, fusion candidates flagged
> **Related Research:** 158 (MUX — closest cousin, vocabulary superposition), 187 (S2F/DeGRPO — gap table SwiR fills), 204 (NFCoT — continuous CoT), 072 (DMax SPD — hybrid embedding cousin), 055 (Tri-Mode Diffusion), 212 (Collapse-Aware Thinking — cousin)
> **Related Plans:** 211 (Three-Mode Router), 212 (Collapse-Aware), 215 (Regime-Transition — shipped, explicit-token only), 109 (DMax SPD), 194 (thinking_cot), 204 (Selectivity Router), 275 (this doc — implementation)
> **Classification:** Public — generic inference engine mechanics
> **PASS-Redirects (synthesis):** Huang, Ma & Liu [alphaXiv `2611.feedback-shortcut-speculative-decoding` "Feedback Shortcut: Speculative Decoding for Latent Feedback Transformers"] (2026-11-20, ifm-ai/LoRM) — the latent-feedback family's serving-class fix (stand-in states + pipelined correction + a D-commit gate; 2.75× vs 2.93× vanilla at rounding-level quality, not exact). SwiR link: SwiR's modelless soft-embedding feedback creates the same position-chain dependency, so the pipelined-correction half applies TRAINING-FREE (stand-in = the last committed soft embedding) — the cheapest route from that paper to a live stack consumer. Recorded riir-infer R037 §2/§5; no consumer exists (specdec paused, riir-infer d9c2bb28) — lane intel only.
> **PASS-Redirects (synthesis):** Wang et al. [arXiv:2608.08888 "Full-bandwidth transformer"] — latent feedback decoding (GLU fusion `W_U h_{t-1} ⊙ σ(W_G e_t)` feeding the previous top-layer hidden state back as the next transformer input) is the TRAINED alternative to SwiR's training-free soft-embedding feedback `ẽ_t = Σ_v p_t[v]·e(v)`. Paper explicitly states latent feedback "cannot simply be switched on at inference" without multi-pass scheduled pretraining (progressive schedule + prefix mixin + jitter noise + depth scaling). §3.5 Path 0 fails: the GLU fusion math is modelless, but the model's inability to process hidden-state-space inputs is a missing capability, not a snapshot/LoRA/latent-projection-correctable systematic bias. Paths 1–3 fail. Pretraining recipe (1B model, 400B tokens) → **riir-train** but beyond current training scope (C13/C14 0.4B from-scratch, Gemma-2 rank-16 LoRA). Runtime mechanism already shipped modellessly: SwiR soft embedding (this note, DEFAULT-ON) + LT2 looped weight-sharing (Research 073, Plan 108) + Training-Free Loop stack looping (Research 097, Plan 136). NPC cognition already has full-bandwidth feedback at the belief level (HLA `evolve_belief` carries the full 8-dim state forward across ticks). Same PASS pattern as Ouro / System-1.5 / ELT / MoR / "Towards Looped Models Done Right" (training-only latent reasoning → PASS; modelless runtime analog shipped).
> **PASS-Redirects (synthesis):** Xiong, Chen & Chen [blog 2026-11-16 "Don't feed back the last block: the best source we found sits just below the top" (https://xjxyys.github.io/blog/feedback-source/)] — the latent-feedback family's source-depth follow-up: a BELOW-TOP source block (7/8, 8/10, ~0.80–0.875 relative depth) beats the last block by −4.4/−4.1e−3 nats across two Marin-MoE sizes (free: equal params/FLOPs), and additive zero-init feedback (`e_t + W·RMSNorm(h)`) beats FBT's gated replacement by −5.9e−3 at the same last-block source (gating+normalization confound disclosed) — the top state is readout-specialized for the token the next position already holds as an embedding. Training-gated (the Path-0 fail above stands); recipe inputs → riir-train Plan 344 / Issue 690 (2026-11-19). SwiR soft-embedding unchanged — no source-depth axis in token space.

---

## TL;DR

SwiReasoning (SwiR) is a **training-free** reasoning framework that **dynamically alternates between explicit (token-based) CoT and latent (soft-embedding-mixture) reasoning** based on block-wise confidence derived from next-token entropy trends. Three mechanisms compose: (1) block-relative entropy switch (H_t vs block-start reference H̄), (2) asymmetric dwell windows (W_E→L = 512, W_L→E = 0 — explicit needs consolidation time, latent exits immediately on confidence recovery), (3) switch count controller with convergence trigger (force `</think>` at ½C_max) and termination trigger (force answer prefix at C > C_max, allow B more tokens). Results: **+1.8–3.1% accuracy** across math/STEM/coding/general reasoning, **+57–79% token efficiency** under limited budgets, **1.36× fewer TFLOPs and 1.36× faster wall-clock** at 90% accuracy, peak efficiency gains of **4.6–6.8× over CoT**.

**Distilled for katgpt-rs (modelless, inference-time):**

Three transferable primitives, all inference-time, no training:

1. **Block-Relative Entropy Switch** — `mode_{t+1} = Explicit if (H_t < H̄) else Latent if (H_t > H̄ ∧ Δt ≥ W_E→L) else mode_t`. Resets H̄ ← H_t on every switch. Unlike our `SelectivityRouter` (static EMA kurtosis) and `CollapseClassifier` (binary post-collapse), this is a **per-block relative signal** — it detects rising/falling confidence within the current thinking block, not against a global baseline. This is the missing piece from Research 187's gap table: "RiM slots: no signal to stop thinking mid-reasoning" → SwiR provides exactly that signal.
2. **Asymmetric Dwell Windows** — Explicit→Latent requires W_E→L steps of sustained uncertainty (default 512) to avoid oscillation; Latent→Explicit fires immediately (W_L→E = 0) on confidence recovery to prevent spurious signal injection. Ablation: removing the window drops avg accuracy 89.10% → 91.92% (−2.82pp); symmetric windows underperform asymmetric at all sizes. This is novel relative to our `RegimeTransitionGate` (Plan 215), which has no dwell concept.
3. **Switch Count Controller** — caps Latent→Explicit transitions at C_max, with convergence trigger at ½C_max (force `</think>` token) and termination trigger at C > C_max (inject answer prefix, B-token budget). Ablation: dropping C_max from ∞ to 20 retains 87.80% accuracy (vs 93.00% at ∞) while cutting generation length from 4924 → 3597 tokens. This is the overthinking-suppression mechanism our `thinking_cot` and `collapse_aware_thinking` lack — they detect collapse but don't bound total reasoning effort.

The **soft embedding formula** `ẽ_t = Σ_v p_t[v] · e(v)` (probability-weighted mixture over the vocabulary embedding matrix) is identical to Soft Thinking (Zhang et al., 2025) — SwiR's contribution is the switching wrapper around it, not the latent step itself.

---

## 1. Paper Core Findings

### 1.1 Two-Mode Taxonomy

| Mode | Policy | Per-step cost | Information preserved | Risk |
|------|--------|---------------|----------------------|------|
| **Explicit** | `x_t ~ π_t(·)` (argmax or top-k sample) | 1 token commit | Single trajectory, distribution collapsed | Discards uncertainty, may over-exploit |
| **Latent** | `ẽ_t = Σ_v p_t[v] · e(v)` (soft mixture) | 1 embedding fed back | Full first-order distribution | Diffuses probability mass, drifts, slower convergence |

Pure latent (Soft Thinking) underperforms CoT by −1.46pp avg (Tab. 1) because sustained latent mode spreads mass across many implicit paths, introducing noise that hurts final accuracy. Pure explicit over-commits and overthinks. SwiR's thesis: **switch modes based on confidence**, exploiting latent's exploration when uncertain and explicit's consolidation when confident.

### 1.2 Block-Relative Entropy Switch

Reference entropy `H̄` is initialized at the first step of each thinking block and refreshed on every mode switch. Decisions:

```
Latent → Explicit : (H_t < H̄)                      # confidence rising, consolidate
Explicit → Latent : (H_t > H̄) ∧ (Δt ≥ W_E→L)       # confidence dropping, dwell first
```

Reset `H̄ ← H_t`, `Δt ← 0` on every switch. Otherwise `Δt ← Δt + 1`. The asymmetry is load-bearing: latent is divergent ( exploration), so it should exit immediately once confidence recovers to avoid injecting spurious signals; explicit is convergent (chain extension), so it needs time to stabilize before a single entropy fluctuation can flip it back.

### 1.3 Switch Count Controller

`C_t` counts completed Latent→Explicit switches. Two triggers, both fire on Latent→Explicit transitions:

- **Convergence trigger** `(½C_max ≤ C_t ≤ C_max)`: enqueue `[ID(</think>)]`. Encourages (not enforces) the model to start converging to an answer based on partial reasoning trajectories.
- **Termination trigger** `(C_t > C_max)`: enqueue `["</think>\n\nThe final answer is"]`, start budget counter `b_t = B`, decrement per step, terminate at `b_t = 0`.

Triggers overwrite the model's next-token output via a per-sample injection queue `Q_t`. C_max is the user-facing budget knob: smaller C_max → earlier answers, lower accuracy but higher token efficiency.

### 1.4 Thinking-Related Signal Mixing

At switch instants, bias the first step's embedding toward thinking-control tokens to align with the model's trained reasoning patterns:

```
Latent entry (step t*):  ẽ_{t*} ← α_{t*} · ẽ_{t*} + (1 - α_{t*}) · e_<think>
Explicit exit  (step t†): ẽ_{t†} ← β_{t†} · ẽ_{t†} + (1 - β_{t†}) · e_</think>
```

Schedules: `α_t = α_0 + (1 - α_0) · t/T_max`, `β_t = β_0 + (1 - β_0) · t/T_max`. Best `β_0 = 0.7` (Tab. 2); `α_0` exposed to user (broad plateau 0.4–0.9, peak at 1.0). Signal mixing contributes +0.6pp avg (Tab. 9).

### 1.5 Empirical Headlines

| Metric | Result | vs |
|--------|--------|-----|
| Avg accuracy (unlimited budget) | +1.8 to +3.1 pp | CoT, CoT-greedy, Soft Thinking |
| AIME 2024/2025 gains | +3.34/+2.50 pp (Qwen3-8B), +5.00/+5.00 pp (Qwen3-1.7B) | CoT |
| Token efficiency (limited budget) | +57% to +79% | CoT |
| Peak efficiency gain | 4.6× to 6.8× | CoT |
| TFLOPs at 90% accuracy | 1.36× fewer | CoT |
| Wall-clock at 90% accuracy | 1.36× faster | CoT |
| Wall-clock at 80% accuracy | 2.17× faster | CoT |
| Pass@k peak sample count | 27–72% fewer | CoT |
| Coding hard-level (MBPP) | +18.18 pp | CoT |

### 1.6 Where SwiR Fails

3D surface shortest-path problems with rigid topological constraints (Appendix C.4): latent mode's smoothing blurs the "only walk across the ceiling and the walls" constraint, producing invalid Euclidean paths. Lesson: **latent exploration is detrimental for rigid geometric constraint satisfaction**. This is exactly the kind of task that should be force-routed to explicit-only mode by an outer controller.

---

## 2. Distillation

### 2.1 Direct Mapping (modelless, katgpt-rs)

Three modules under `src/swir/`:

**M1. `SwiRController` (state machine):**

```rust
pub struct SwiRController {
    mode: ThinkMode,            // Explicit | Latent
    reference_entropy: f32,     // H̄ — block-start entropy, reset on switch
    dwell_steps: u32,           // Δt — steps since last switch
    switch_count: u32,          // C_t — completed Latent→Explicit transitions
    // Config (SwiRConfig):
    w_e_to_l: u32,              // default 512
    w_l_to_e: u32,              // default 0
    c_max: u32,                 // user-facing budget knob
    c_convergence_fraction: f32,// default 0.5
    answer_budget_b: u32,       // tokens allowed after termination trigger
    alpha_0: f32,               // latent-entry signal mixing ratio
    beta_0: f32,                // explicit-exit signal mixing ratio, default 0.7
}

pub enum ThinkMode { Explicit, Latent }
pub enum StepAction {
    EmitToken(u32),             // explicit mode: argmax or sampled token id
    EmitSoftEmbedding(/* ẽ_t buffer */),  // latent mode: probability-weighted mixture
    InjectTokens(Vec<u32>),     // convergence/termination trigger fired
    Terminate,                  // termination trigger + budget exhausted
}

impl SwiRController {
    /// Called per decode step. entropy = H_t computed from softmax(logits).
    /// Returns the action to take and updates internal state.
    pub fn step(&mut self, entropy: f32, step_index: u32, max_steps: u32) -> StepAction { ... }
}
```

**M2. Soft embedding computation (latent mode):**

```rust
/// ẽ_t = Σ_v p_t[v] · e(v) — probability-weighted mixture over embedding matrix.
/// Zero-allocation: caller passes scratch buffer of size embedding_dim.
/// SIMD-friendly: chunked 8-wide dot products.
pub fn soft_embedding(
    probs: &[f32],              // p_t, length = vocab_size
    embedding_matrix: &[f32],   // flattened row-major, vocab_size × embedding_dim
    embedding_dim: usize,
    scratch: &mut [f32],        // length = embedding_dim, zeroed on entry
);
```

Already have similar infrastructure in `crates/katgpt-sparse/src/sparse_compose.rs` and `crates/katgpt-core/src/mux_demux.rs` (Research 158). The soft-embedding op is the same primitive as MUX's `mux(r_i) = Σ_j w_j · onehot(token_j) / Z`, just with `w_j = p_t[v]` (continuous probabilities) instead of geometric decay weights.

**M3. Signal mixing at switch instants:**

```rust
/// Blends ẽ with control-token embedding at mode transitions.
/// α_t = α_0 + (1 - α_0) · t / T_max
pub fn mix_thinking_signal(
    soft_embed: &mut [f32],
    control_token_embed: &[f32],   // e_<think> or e_</think>
    alpha_0: f32,
    step_index: u32,
    max_steps: u32,
);
```

### 2.2 Integration with Existing Infrastructure

SwiR is a **mode-switching controller**, not a complete reasoning pipeline. It plugs into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`:

```
thinking_cot (Plan 194)
    ├── SelectivityRouter (Plan 204)      — kurtosis-based direct-vs-CoT pre-decide
    ├── ThinkingController (bandit)       — global thinking budget
    ├── CollapseAwareThinking (Plan 212) — mid-reasoning early-exit on collapse
    └── SwiRController (NEW, Plan 275)    — explicit↔latent mode switch + count cap
```

The four are complementary, not competing:
- `SelectivityRouter`: **before** thinking (do I think at all?)
- `ThinkingController`: **globally** (how much total budget?)
- `CollapseAwareThinking`: **on collapse** (stop early because of failure signal)
- `SwiRController`: **during** thinking (which mode am I in right now, and have I switched too many times?)

Latent mode reuses `rim_slots` (Plan 172) infrastructure for the soft-embedding workspace. Explicit mode is the standard argmax/sample path.

### 2.3 Fusion (Super-GOAT candidates for future research notes)

Per skill §1: a fusion produces a new capability class when no incumbent can do it. Three fusion candidates flagged here; each warrants its own research note if pursued.

**Fusion A — Sub-Token-Resolution Continuous-Mode Router** (SwiR × Plan 215 Regime Transition × Plan 109 DMax SPD):

SwiR uses binary mode switch with signal mixing only at switch instants. DMax SPD uses hybrid embedding `h = conf · e_token + (1 − conf) · e_mask` for dllm diffusion. **Fuse:** replace SwiR's binary switch with a **sigmoid-weighted blend** `ẽ_t = σ(λ · (H̄ − H_t)) · ẽ_latent + (1 − σ(...)) · e_argmax_token`, gated by entropy trend. The closer confidence is to rising, the more weight on explicit token; the closer to dropping, the more on latent mixture. This creates a **continuous mode router** at sub-token resolution — no incumbent (215, 211, 204, 212) has this. **Novelty gate:** Q1 ✓ (no prior art for the combination), Q2 ✓ (new capability class: continuous-mode reasoning), Q3 ✓ (selling point: "smooth interpolation between token and latent reasoning at sub-token resolution"), Q4 ✓ (connects 215 + 211 + 109 + 172). **All 4 YES → Super-GOAT candidate.** Validation: Pareto curve vs binary SwiR on MATH500/AIME.

**Fusion B — SwiR × MUX Vocabulary Superposition** (SwiR × Research 158 × Plan 211 Three-Mode Router):

MUX's `mux(r_i) = Σ_j w_j · onehot(token_j) / Z` with geometric decay `0.9^j` is structurally identical to SwiR's `ẽ_t = Σ_v p_t[v] · e(v)` — both are points in the vocabulary simplex representing multiple tokens. **Fuse:** add a **Latent arm** to Plan 211's 6-arm UCB1 bandit (currently all arms are explicit-token regimes: L4R, R4L, LR × 2 verification tiers). The bandit learns when to invoke SwiR's latent mode vs explicit-token modes based on per-arm reward (downstream task success). **Novelty gate:** Q1 ✓, Q2 ✓ (first bandit-router with explicit latent arm), Q3 ✓, Q4 ✓ (211 + 158 + 194 + 272). **Super-GOAT candidate.**

**Fusion C — SwiR for NPC Think-Brain/Info-Brain Cycling** (SwiR × riir-ai two-brain spatial cognition model):

Per AGENTS.md latent-vs-raw rules: info brain (real `MapPos`, synced) vs think brain (`SpatialBelief`, fog-of-war, not synced). **Map:** SwiR's Explicit mode = info brain commit (NPC commits to a concrete action with raw coordinates). SwiR's Latent mode = think brain exploration (NPC considers multiple action hypotheses in policy-latent space without committing). Switch count controller = NPC's "thinking budget" before forced action — bounded deliberation prevents analysis paralysis in combat. Entropy trend = action-distribution uncertainty (high entropy = many plausible actions = explore in latent; low entropy = committed action = emit raw). **This is the missing runtime signal for the think→info bridge** — currently the bridge is one-way (info→think on visibility), with no mechanism for think→info commit triggers. SwiR's entropy-trend switch provides exactly that trigger. **Routing:** private guide → `riir-ai/.research/`, NOT katgpt-rs (game IP). Open primitive stays in katgpt-rs.

### 2.4 What NOT to Distill

- **Token injection queue mechanics** — paper's `Q_t` queue that overwrites next-token output is implementation detail; we already have injection patterns in `llmexec_guard` (Plan 223) and can reuse.
- **Hyperparameter tables per benchmark** (Tab. 6) — `α_0` is benchmark-dependent and user-exposed. Ship one good default (`α_0 = 0.6, β_0 = 0.7, W_E→L = 512, W_L→E = 0, C_max = 20`), expose the rest via `SwiRConfig`.
- **Pass@k evaluation** — interesting but not a distillation target; it's a benchmarking protocol.

---

## 3. Verdict

**Tier: GOAT** — provable gain (latency + quality), not a new capability class on its own (Soft Thinking is direct prior art for the latent step; SwiR's contribution is the switching wrapper).

**One-line reasoning:** SwiR's three mechanisms (block-relative entropy switch + asymmetric dwell + switch count controller) compose into a training-free controller that fills the exact gaps Research 187 identified in our thinking-cot stack ("no signal to stop thinking mid-reasoning", "no per-instance early exit", "resamples from same distribution — no mode switch"). Paper reports +1.8–3.1pp accuracy and 1.36–6.8× efficiency gains, plug-and-play at inference time.

**Routing:** Plan only in `katgpt-rs/.plans/275_swir_switch_thinking.md`. Feature flag `swir_switch_thinking`, default-off until GOAT proof. No riir-ai guide (per skill: GOAT verdict → plan only, no guide). Fusion C (NPC think-brain) flagged as future riir-ai research if Super-GOAT validation passes. *(Post-promotion update: Plan 313 T6.2 GOAT G2 token-efficiency PASS; `swir_switch_thinking` is now DEFAULT-ON in root Cargo.toml default array.)*

**GOAT gate (must pass before promoting to default):**
- G1: ≥ +1.5pp avg accuracy on internal reasoning benchmark vs `thinking_cot` baseline (paper reports +2.17pp avg on Qwen3-8B).
- G2: ≥ 1.3× token efficiency at fixed accuracy (paper reports 1.36× at 90% accuracy).
- G3: Zero-allocation hot path — `SwiRController::step()` must complete in < 200ns (entropy compute + state update + decision).
- G4: Mode-switch correctness — verify soft-embedding output lies in vocabulary convex hull (Lyapunov-style invariant; `min_v e(v) ≤ ẽ_t ≤ max_v e(v)` componentwise).
- G5: No regression on `thinking_cot` and `collapse_aware_thinking` tests when `swir_switch_thinking` is disabled.
- G6: Failure mode handling — 3D-surface-shortest-path-style tasks (Appendix C.4) should auto-fall-back to explicit-only mode via `SelectivityRouter` kurtosis signal.

If G1–G6 pass → promote to default. If G1 fails but G2/G3 pass → keep as opt-in efficiency feature. If G3 fails → demote, investigate SIMD soft-embedding kernel.

**Fusion follow-ups (separate research notes, not blocking):**
- Fusion A (sub-token continuous router) — Super-GOAT candidate, needs Pareto proof vs binary SwiR.
- Fusion B (MUX × SwiR bandit arm) — Super-GOAT candidate, needs bandit convergence proof.
- Fusion C (NPC two-brain) — riir-ai guide only after Fusion A validates the core primitive.

---

## 4. Cross-References

| Research | Connection |
|----------|------------|
| 158 (MUX) | Closest cousin — vocabulary superposition. SwiR's `ẽ_t` is structurally identical to `mux(r_i)`. Fusion B. |
| 187 (S2F/DeGRPO) | Gap table L85-91 is filled point-by-point by SwiR. Direct complement. |
| 204 (NFCoT) | Continuous CoT via normalizing flow — different mechanism, same goal. |
| 072 (DMax SPD) | Hybrid embedding for dllm — Fusion A uses DMax's `h = conf · e + (1−conf) · e_mask` as the continuous-switch bridge. |
| 055 (Tri-Mode) | AR + Diffusion + Self-Spec — orthogonal axis (decode mode vs reasoning mode). |
| 212 (Collapse-Aware) | Cousin — collapse detection vs entropy-trend switching. Compose, don't replace. |
| 211 (Three-Mode Router) | Fusion B target — add latent arm to bandit. |
| 215 (Regime-Transition) | Already-shipped cousin — routes in explicit-token space only; SwiR adds latent arm. |

| Plan | Connection |
|------|------------|
| 194 (thinking_cot) | **Integration target** — SwiR plugs in as a new `ThinkingStrategy`. |
| 211 (Three-Mode Router) | Fusion B — bandit arm extension. |
| 212 (Collapse-Aware) | Compose — collapse triggers exit, SwiR controls mid-thinking mode. |
| 215 (Regime-Transition) | Reference for GOAT gate methodology (8/8 mock + 4/4 real). |
| 109 (DMax SPD) | Fusion A — borrow hybrid embedding formula. |
| 172 (RiM slots) | Reuse for latent-mode soft-embedding workspace. |
| 204 (Selectivity Router) | Compose — kurtosis decides if we think at all; SwiR decides how. |
| 275 (this distillation's plan) | Implementation. |

---

## TL;DR

SwiReasoning (ICLR 2026, arXiv:2510.05069) is a **training-free** explicit↔latent reasoning switch with three mechanisms: block-relative entropy trend (vs reference H̄), asymmetric dwell windows (W_E→L=512, W_L→E=0), and switch count controller (convergence at ½C_max, termination at C>C_max). Reports +1.8–3.1pp accuracy, 1.36–6.8× efficiency, plug-and-play at inference. **Verdict: GOAT** — Soft Thinking is direct prior art for the latent step, so fails Super-GOAT Q1 (prior art). Direct mapping: three modules under `src/swir/` (`SwiRController`, `soft_embedding`, `mix_thinking_signal`), feature flag `swir_switch_thinking`, integrates into `thinking_cot` (Plan 194) as a new `ThinkingStrategy`. Fills Research 187's gap table point-by-point. **Three Super-GOAT fusion candidates flagged** for separate research notes: (A) sub-token continuous-mode router via DMax SPD hybrid embedding, (B) MUX vocabulary superposition as a latent arm in Plan 211's bandit, (C) NPC think-brain/info-brain cycling trigger in riir-ai. Plan: `katgpt-rs/.plans/275_swir_switch_thinking.md`. GOAT gate G1–G6 must pass before promoting to default.
