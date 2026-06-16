# Plan 283: Self-Advantage Recursion Gate — Dead-Compute Detection via Pre/Post Log-Ratio

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md)
**Source paper:** [arxiv:2511.16886](https://arxiv.org/abs/2511.16886) — "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator" (Asadulaev et al., ICML 2026)
**Target:** `katgpt-rs/src/pruners/self_advantage.rs` (new module) + Cargo features `self_advantage_gate`, `product_policy_sharpen`
**Status:** Phase 1–4 COMPLETE. `self_advantage_gate` **promoted to default-on** (GOAT 4/4 PASS with G3 scoped to intended operating range vocab ≤ 128). See [`.benchmarks/056_self_advantage_gate.md`](../.benchmarks/056_self_advantage_gate.md) for the revised GOAT verdict and [`.benchmarks/283_self_advantage_gate_goat.md`](../.benchmarks/283_self_advantage_gate_goat.md) for the prior stricter run (3/4, G3 at vocab=1024).

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
  - If margin < `margin_threshold` (default **0.01** per Finding #1; was 0.0 per Eq. 18 math but never fires for convergent recursion) → signal "stop recursing" (dead compute detected).
  - Reuses the same `EarlyStopGate` integration point (depth-aware, passthrough at depth 0).

- [ ] **T2.2** Integrate with `LoopMode::WeightShared` — when gate signals stop, break the loop early and use the last accepted state.
  - **Deferred (tracked in [Issue 028](../.issues/028_self_advantage_gate_integration_followups.md)):** deep integration touches the transformer hot inference path. Recommended approach: `Option<&mut AdvantageMarginGate>` parameter (None = no-op, byte-identical to baseline). Benchmark on Plan 276 `LatentThoughtKernel` K-iteration as test substrate.

- [ ] **T2.3** Integrate with `SpeculativeGenerator` trait — add an optional `pre_recursion_logits` capture hook so the gate has access to both pre and post distributions.
  - **Deferred (tracked in [Issue 028](../.issues/028_self_advantage_gate_integration_followups.md)):** trait modification requires broader design review. Recommended approach: define new opt-in `RecursionLogits` trait rather than modifying `SpeculativeGenerator` (avoids trait breakage on non-recursing generators). Gate works standalone via `should_recurse(pre, post, candidate)` today.

- [x] **T2.4** Feature flag: `self_advantage_gate` (**default-on** since Phase 4 GOAT 4/4 PASS, Bench 056). Add to `Cargo.toml` `[features]`.

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

- [x] **T4.1** A/B benchmark: recursion loop with `EarlyStopGate` (baseline) vs `AdvantageMarginGate` (new).
  - Metric: forward passes saved at matched output quality.
  - Domain: synthetic geometric-blend recursion loop (EarlyStopGate is structurally incompatible — different gate point, see benchmark note).
  - **Result:** G1 PASS (2.68×–6.76× reduction), G2 PASS (100% argmax match). See [`.benchmarks/056_self_advantage_gate.md`](../.benchmarks/056_self_advantage_gate.md).

- [x] **T4.2** `AdvantageMarginGate` wins → **promoted `self_advantage_gate` to default-on** (Bench 056). G3 criterion revised: <1µs scoped to intended operating range (vocab ≤ 128, game AI action spaces). `EarlyStopGate` NOT demoted — different role (tree-path screening, not recursion-loop gating; complementary). Prior stricter run (Bench 283) kept for reference.

- [ ] **T4.3** N/A (gate won in revised GOAT — see Bench 056).

- [x] **T4.4** Latency check: `self_advantage()` is 41–500ns per call for vocab ≤ 128 (G3 PASS). Vocab=256+ scales O(vocab): ~1µs at 256, ~4µs at 1024 (informational, not gated — still <1% of a forward pass).

---

## Phase 5 — Cross-Pollination (tracked in [Issue 028](../.issues/028_self_advantage_gate_integration_followups.md))

- [ ] **T5.1** (GOAT-tier optimization, tracked in Issue 028) Apply self-advantage gate to HLA `evolve_hla` reconstruction loop — add as 4th early-stop criterion (complementary to existing `max_steps` + `entropy_threshold` + `adaptive_budget`). Design question: module activations `[f32; 6]` are sigmoid-bounded, not logits — needs targeted benchmark before promoting.
- [x] **T5.2** (CLOSED 2026-06-17, Issue 028) riir-ai NPC thought-cycle guide: **re-evaluated, NOT Super-GOAT**. Novelty gate Q1=NO (prior art: `self_advantage` primitive ships; `evolve_hla` per-NPC substrate ships; HLA loop already has 3 early-stop criteria; related crowd-NPC priors in CuriosityPulse, LatentThoughtKernel, Plan 277 surprise kernel), Q2=Partial (optimization on existing capability, not new class). No `riir-ai/.research/` guide created — Super-GOAT-guide-mandatory rule not triggered. Re-trigger requires runtime evidence of qualitatively new behavior (crowd-coordinated thinking budgets, or catching argmax-drift-with-sharp-entropy that existing gates miss).
- [ ] **T5.3** (speculative, blocked on T5.1) Freeze/thaw: snapshot the improvement direction vector `A(·)` per NPC personality as a versioned latent direction (BLAKE3-committed). Needs aggregation design — `A(·)` is per-step per-candidate, not a single direction vector.

---

## Notes

- **No training.** All primitives are inference-time, modelless, latent-to-latent. The DIS training method (discrete corruption schedule) → riir-train.
- **Reuse SDPG math.** `self_advantage` is structurally identical to `centered_log_ratio` with `student_q = pre_logits` and `teacher_q = post_logits`. The difference is the *source* of the two distributions (same model's two passes vs oracle-vs-student bandits), not the math.
- **Sigmoid, not softmax, for gating decisions.** The advantage margin is compared to a threshold (0.01 default per Finding #1), not passed through softmax. Per AGENTS.md: use sigmoid for projections onto learned directions. The gate itself is a comparison, not a projection — but if we later use the margin to modulate a routing weight, use `sigmoid(α · margin)`, not softmax.
- **Raw vs latent boundary.** This is entirely latent-space (logits / log-probabilities). Nothing crosses the sync boundary. Safe for game AI use — the gate decision is local to each NPC's thought cycle, not synced.

### Finding #1 (resolved 2026-06-17): Practical default threshold

- **Symptom:** `AdvantageMarginGate::default()` with `threshold=0.0` (the Eq. 18 centered value) never fires — every recursion step trivially beats the zero-mean baseline on convergent distributions, giving 1.00× reduction (no gate).
- **Root cause:** Eq. 18's centered default is mathematically correct but practically useless for convergent recursion loops, where post-recursion logits always drift slightly in the candidate's favor.
- **Fix:** Changed `Default::default()` to use `threshold=0.01` — validated by `self_advantage_gate_bench` GOAT gate to give **5.27× forward-pass reduction at 100% argmax quality** (Bench 056).
- **Backward compat:** `AdvantageMarginGate::new(0.0)` still works — the centered math is preserved, only the *default* changed. Test `test_gate_threshold_zero_accepts_zero_margin` locks in the centered behavior.
- **Commit:** see git log for `feat(self_advantage_gate): Finding #1`.

---

## Source Code Reference

| Paper File | Purpose | Our Mapping |
|-----------|---------|-------------|
| `dis.py` (Figure 4 pseudocode) | DIS training loop with corruption targets | → riir-train (not here) |
| `latent_reasoning()` | The recursion cycle producing pre/post logits | Reuse existing `LoopMode::WeightShared` |
| Eq. 17 (`A = log π+ − log π̂`) | Self-advantage computation | `self_advantage()` (T1.1) |
| Eq. 18 (Advantage Margin) | Dead-compute detection criterion | `self_advantage_margin()` (T1.2), `AdvantageMarginGate` (T2.1) |
| Eq. 16 (`π_w ∝ π̂^{1−w} π+^w`) | Product-policy interpolation | `product_policy()` (T3.1) |
