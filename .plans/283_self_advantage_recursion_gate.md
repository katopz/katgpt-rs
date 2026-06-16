# Plan 283: Self-Advantage Recursion Gate — Dead-Compute Detection via Pre/Post Log-Ratio

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md)
**Source paper:** [arxiv:2511.16886](https://arxiv.org/abs/2511.16886) — "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator" (Asadulaev et al., ICML 2026)
**Target:** `katgpt-rs/src/pruners/self_advantage.rs` (new module) + Cargo features `self_advantage_gate`, `product_policy_sharpen`
**Status:** Active — Phase 1 + 2 + 3 COMPLETE (math + wrappers + examples). Phase 4 (GOAT gate benchmark) pending.

---

## Goal

Ship three modelless primitives derived from the policy-improvement theoretical lens on latent recursion:

1. **`self_advantage()`** — compute the log-ratio `A(a) = log π+(a) − log π̂(a)` between a model's pre-recursion and post-recursion logits. No teacher, no oracle. Reuses SDPG's `centered_log_ratio` internal math but sources both distributions from the same model.
2. **`AdvantageMarginGate`** — accept a recursion step iff `A(y*) > E_a[A(a)]` (Eq. 18). Skip dead compute. Paper claims 18× forward pass reduction.
3. **`product_policy()`** — inference-time multiplicative interpolation `π_w ∝ π̂^{1−w} · π+^w` (Eq. 16). Controllable reasoning trust weight `w`.

**GOAT gate:** ≥2× forward pass reduction at matched output quality on HLA belief evolution or DDTree speculative decode. If wins → promote to default; demote `EarlyStopGate` if it loses on the same benchmark.

---

## Phase 0 — Benchmark Baseline (MUST DO FIRST)

### Tasks

- [ ] **T0.1** Identify benchmark domain — HLA belief evolution on bomber arena (per Plan 180 precedent) OR DDTree speculative decode. Pick the one with an existing loop/recursion to instrument.
- [ ] **T0.2** Record baseline: forward passes per inference, output quality metric (accuracy / acceptance rate), latency per token/decision.
- [ ] **T0.3** Instrument pre/post recursion logits capture — add temporary logging hooks at the recursion boundary (do not ship this; measurement only).

---

## Phase 1 — Self-Advantage Computation (~150 LOC, `self_advantage.rs`)

### Tasks

- [x] **T1.1** `fn self_advantage(pre_logits: &[f32], post_logits: &[f32], scratch: &mut [f32]) -> &[f32]`
  - Returns `A(a) = post_logsoftmax[a] − pre_logsoftmax[a]` per action.
  - Zero-allocation: writes into caller-provided `scratch` buffer (same length as logits).
  - SIMD-friendly: chunked loop over 4 or 8 elements for auto-vectorization.
  - Numerical stability: use log-softmax (not raw log of softmax) to avoid overflow — compute max-subtracted logits first.

- [x] **T1.2** `fn self_advantage_margin(pre_logits: &[f32], post_logits: &[f32], candidate: usize, scratch: &mut [f32]) -> f32`
  - Returns the **Advantage Margin** for a specific candidate action: `A(candidate) − E_{a∼π_w}[A(a)]`.
  - The expectation is computed under the interpolated policy `π_w` with `w=1.0` (post-recursion policy) as the default weighting.
  - Positive margin = this candidate benefits from the recursion step. Negative = dead compute for this candidate.

- [x] **T1.3** Unit tests
  - Identical pre/post logits → all advantages zero, margin zero (dead compute correctly detected).
  - Post-recursion sharpens toward candidate → positive margin for candidate.
  - Post-recursion shifts away from candidate → negative margin (correctly flagged as harmful step).
  - Numerical stability: extreme logits (±1e4) don't overflow.

- [x] **T1.4** Property test: `self_advantage` recovers SDPG's `centered_log_ratio` up to a state-dependent constant when pre=student and post=teacher. (Cross-validation with Plan 180's shipped implementation.)

---

## Phase 2 — AdvantageMarginGate (~120 LOC)

### Tasks

- [x] **T2.1** `pub struct AdvantageMarginGate<P> { inner: P, margin_threshold: f32, scratch: Vec<f32> }`
  - Wraps any pruner/generator that produces logits.
  - After each recursion step, computes `self_advantage_margin()`.
  - If margin < `margin_threshold` (default 0.0) → signal "stop recursing" (dead compute detected).
  - Reuses the same `EarlyStopGate` integration point (depth-aware, passthrough at depth 0).

- [ ] **T2.2** Integrate with `LoopMode::WeightShared` — when gate signals stop, break the loop early and use the last accepted state.
  - **Deferred:** deep integration requires modifying transformer forward pass. Gate is usable standalone via `should_recurse()`.

- [ ] **T2.3** Integrate with `SpeculativeGenerator` trait — add an optional `pre_recursion_logits` capture hook so the gate has access to both pre and post distributions.
  - **Deferred:** trait modification requires broader design review. Gate works with any logits source via `should_recurse(pre, post, candidate)`.

- [x] **T2.4** Feature flag: `self_advantage_gate` (opt-in, off by default). Add to `Cargo.toml` `[features]`.

- [x] **T2.5** Example: `examples/pruner_03_self_advantage_gate.rs` — demonstrate dead-compute detection on a synthetic recursion loop. Show forward passes saved.

---

## Phase 3 — Product-Policy Sharpening (~80 LOC)

### Tasks

- [x] **T3.1** `fn product_policy(pre_logits: &[f32], post_logits: &[f32], w: f32, out: &mut [f32])`
  - Returns log-space interpolation: `out[a] = (1−w) · pre_logsoftmax[a] + w · post_logsoftmax[a]`.
  - Caller softmaxes the result to get `π_w`.
  - `w=0.0` → pre-recursion (skip reasoning). `w=1.0` → post-recursion (full reasoning). `w>1.0` → extrapolation (trust reasoning beyond the model's own update).
  - Zero-allocation, writes to caller buffer.

- [x] **T3.2** `ProductPolicySharpen<P> { inner: P, w: f32 }` — wrapper that applies product-policy after each recursion step, producing a controllably-sharpened distribution.
  - Implemented as `ProductPolicySharpen { w: f32 }` (no inner generic needed — pure function wrapper).

- [x] **T3.3** Feature flag: `product_policy_sharpen` (opt-in). Add to `Cargo.toml`.

- [x] **T3.4** Example: `examples/pruner_04_product_policy.rs` — sweep `w ∈ {0.0, 0.5, 1.0, 1.5, 2.0}` on a recursion loop, show quality vs compute tradeoff curve.

---

## Phase 4 — GOAT Gate Benchmark

### Tasks

- [ ] **T4.1** A/B benchmark: recursion loop with `EarlyStopGate` (baseline) vs `AdvantageMarginGate` (new).
  - Metric: forward passes saved at matched output quality.
  - Domain: HLA belief evolution OR DDTree speculative decode (whichever T0.1 picked).
  - **Gate:** ≥2× forward pass reduction with no quality loss → promote to default.

- [ ] **T4.2** If `AdvantageMarginGate` wins: demote `EarlyStopGate` to opt-in, promote `self_advantage_gate` to default-on. Update README GOAT table.

- [ ] **T4.3** If `AdvantageMarginGate` loses: keep opt-in, document why in benchmark note (`.benchmarks/NNN_self_advantage_gate.md`). The product-policy sharpening primitive may still ship as opt-in.

- [ ] **T4.4** Latency check: `self_advantage()` must be <1µs per call (O(vocab) SIMD loop). Verify with criterion bench.

---

## Phase 5 — Cross-Pollination (future, not blocking)

- [ ] **T5.1** (future) Apply self-advantage gate to HLA `evolve_hla` — gate which belief updates are worth keeping. (Research 250 §Fusion)
- [ ] **T5.2** (future) riir-ai guide: NPC thought-cycle dead-compute detection at MMORPG scale. Thousands of NPCs × 20Hz tick → skip non-improving thoughts. (Re-evaluate Super-GOAT potential here.)
- [ ] **T5.3** (future) Freeze/thaw: snapshot the improvement direction vector `A(·)` per NPC personality as a versioned latent direction (BLAKE3-committed).

---

## Notes

- **No training.** All primitives are inference-time, modelless, latent-to-latent. The DIS training method (discrete corruption schedule) → riir-train.
- **Reuse SDPG math.** `self_advantage` is structurally identical to `centered_log_ratio` with `student_q = pre_logits` and `teacher_q = post_logits`. The difference is the *source* of the two distributions (same model's two passes vs oracle-vs-student bandits), not the math.
- **Sigmoid, not softmax, for gating decisions.** The advantage margin is compared to a threshold (0.0 default), not passed through softmax. Per AGENTS.md: use sigmoid for projections onto learned directions. The gate itself is a comparison, not a projection — but if we later use the margin to modulate a routing weight, use `sigmoid(α · margin)`, not softmax.
- **Raw vs latent boundary.** This is entirely latent-space (logits / log-probabilities). Nothing crosses the sync boundary. Safe for game AI use — the gate decision is local to each NPC's thought cycle, not synced.

---

## Source Code Reference

| Paper File | Purpose | Our Mapping |
|-----------|---------|-------------|
| `dis.py` (Figure 4 pseudocode) | DIS training loop with corruption targets | → riir-train (not here) |
| `latent_reasoning()` | The recursion cycle producing pre/post logits | Reuse existing `LoopMode::WeightShared` |
| Eq. 17 (`A = log π+ − log π̂`) | Self-advantage computation | `self_advantage()` (T1.1) |
| Eq. 18 (Advantage Margin) | Dead-compute detection criterion | `self_advantage_margin()` (T1.2), `AdvantageMarginGate` (T2.1) |
| Eq. 16 (`π_w ∝ π̂^{1−w} π+^w`) | Product-policy interpolation | `product_policy()` (T3.1) |
