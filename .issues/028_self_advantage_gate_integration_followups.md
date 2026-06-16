# Issue 028: Self-Advantage Gate Integration Follow-ups (Plan 283 Phase 2 Deep + Phase 5)

**Date:** 2026-06-17
**Status:** Open — tracked, **not Super-GOAT** (evaluated 2026-06-17, see §T5.2 verdict)
**Plan:** [283_self_advantage_recursion_gate.md](../.plans/283_self_advantage_recursion_gate.md) — Phase 1–4 COMPLETE, `self_advantage_gate` **default-on** (Bench 056 GOAT 4/4 PASS)
**Research:** [250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md) — original verdict GOAT, Super-GOAT explicitly deferred ("Do NOT pre-claim")
**Primitive:** `katgpt-rs/src/pruners/self_advantage.rs` — `AdvantageMarginGate::should_recurse(pre, post, candidate)`

---

## Context

Plan 283 Phases 1–4 are shipped and GOAT-validated (4/4 green at threshold=0.01: 2.68×–6.76× forward-pass reduction, 100% argmax match, 41–500ns latency for vocab ≤ 128). The gate is a **standalone primitive** consuming `(pre_logits, post_logits, candidate)`. None of the follow-ups below block the shipped primitive — they are integration opportunities that were explicitly deferred in the plan.

Per AGENTS.md ("Create issue at `./issues` for optimization task, do not create plan"), this issue captures the deferred integration items. The T5.2 Super-GOAT re-evaluation has been completed honestly (verdict below): **NOT Super-GOAT**, so no `riir-ai/.research/` guide is created. The Super-GOAT-guide-mandatory rule is not triggered.

---

## T5.2 — riir-ai NPC thought-cycle guide: Super-GOAT re-evaluation (CLOSED)

### The deferred Super-GOAT claim

Research 250 §Verdict: *"If the MMORPG NPC application proves out — thousands of NPCs with per-tick advantage-margin gating saving measurable tick budget — the selling point solidifies. Re-evaluate after Plan implementation + game-side benchmark. Do **not** pre-claim Super-GOAT."*

The plan is now implemented and GOAT-validated. This is the moment to re-evaluate.

### Novelty gate (Q1–Q4, honest assessment)

| Q | Criterion | Answer | Evidence |
|---|-----------|--------|----------|
| Q1 | No prior art? | **NO** | (a) The math primitive (`self_advantage` / `centered_log_ratio`) ships in katgpt-rs (`src/pruners/self_advantage.rs`, `src/pruners/sdpg/advantage.rs`). (b) The per-NPC recurrent belief substrate ships as `ReconstructionState::evolve_hla` (`crates/katgpt-core/src/sense/reconstruction.rs:771`). (c) Existing HLA reconstruction loop already has *three* early-stop mechanisms: `max_steps` (default 3), `entropy_threshold` (default 0.05) via `sufficient()`, and `with_adaptive_budget` (latency-driven). (d) Related crowd-NPC priors: `CuriosityPulse` (riir-ai/.research/041), `LatentThoughtKernel` Family B "K-iteration deliberation ticks" (Plan 276), Plan 277 surprise kernel. The integration is novel, the underlying mechanisms are not. |
| Q2 | New class of behavior? | **Partial** | Adding advantage-margin as a 4th early-stop criterion for HLA reconstruction is an *optimization on an existing capability* (per-NPC adaptive thought budgeting). Not a new capability class. The framing "NPCs autonomously discover when their own thinking has converged" is philosophical restatement of "skip dead HLA steps", not a new behavior. |
| Q3 | Product selling point? | **Uncertain** | "Our NPCs think only when thinking improves them" is a measurable forward-pass reduction claim (paper: 18×; our gate: 2.68×–6.76× at vocab ≤ 128). Selling-point strength depends on customers caring about per-NPC think budget at MMORPG scale — itself unproven in production. |
| Q4 | Force multiplier? | **YES** | Connects ≥5 pillars: SDPG (math), EqR (recursion), HLA `evolve_hla`, CGSP curiosity (riir-ai/041, 126, 127), Plan 276 `MicroRecurrentBeliefState`, Plan 277 `TemporalDerivativeKernel`, Plan 255 ANE batch. |

### Verdict: **NOT Super-GOAT** (Q1 NO, Q2 Partial)

The Super-GOAT-guide-mandatory rule is **not triggered**. No `riir-ai/.research/` guide is created. This matches the canonical precedent: riir-ai/.research/127 (Implicit Microcognition Crowd NPC) was similarly downgraded Super-GOAT → GOAT after a prior-art check on `evolve_hla` (same root cause — the per-NPC substrate already ships).

**Honest tier:** GOAT-tier optimization opportunity (measurable win, no new capability). Tracked as the T5.1 sub-issue below. Re-evaluation is possible if game-side integration proves a *qualitatively* new behavior (not just a measurable speedup) — but that requires runtime evidence we don't have.

### What "Super-GOAT re-trigger" would actually require

To upgrade T5.2 in the future, the game-side benchmark must show one of:

1. **A new emergent behavior class** (not just a speedup) — e.g., NPCs that *coordinate* their thinking budgets across a crowd (one NPC's halt signal cascades to neighbors via CGSP), producing crowd-scale attention patterns no per-NPC gate can produce alone.
2. **A measurable selling point that no competitor matches** — e.g., "we run 10k NPCs at 20Hz on commodity hardware because only the N% with improving thoughts spend the full think budget", backed by a published benchmark.
3. **A capability that the existing 3 early-stop mechanisms cannot achieve** — e.g., the gate catches "this step improved the wrong candidate" cases that entropy-threshold misses (entropy stays sharp but the argmax drifts).

None of these are claimable today. All require runtime integration + benchmark.

---

## T5.1 — Apply gate to HLA `evolve_hla` reconstruction loop

**Integration site:** `ReconstructionState::reconstruct_inner` / `reconstruct_matvec` / `reconstruct_with_weights` (`crates/katgpt-core/src/sense/reconstruction.rs:850-972`)

### Current loop

```text
loop {
    let activations = self.expand_*(brain);          // [f32; 6] — module activations
    let selected = self.route(&activations);          // [bool; 6]
    self.accumulate(&selected, &activations);         // TripleEvidence merge
    self.evolve_hla();                                // leaky-integrator update of [f32; 8]
    self.step += 1;
    if self.sufficient() { return activations; }      // max_steps OR entropy < threshold
}
```

### Existing early-stop criteria (3 — all already shipped)

| Criterion | Signal | Source | Semantics |
|-----------|--------|--------|-----------|
| `max_steps` (default 3) | step count | `ReconstructionConfig` | "MRAgent shows diminishing returns after 3-4" |
| `entropy_threshold` (default 0.05) | activation entropy | `TripleEvidence::activation_entropy()` via `sufficient()` | "evidence is sharp enough — distribution converged" |
| `with_adaptive_budget` (Phase 6) | measured cycle latency | `LATENCY_BUDGET_NS = 500` | "we're spending too much time — reduce max_steps" |

### What advantage-margin would add (4th criterion, complementary)

| Criterion | Signal | Semantics | Catches what the others miss? |
|-----------|--------|-----------|-------------------------------|
| `AdvantageMarginGate` | `A(candidate) − E_a[A(a)]` on routed distribution | "this step did not improve the prediction for the candidate" | **YES** — entropy-threshold catches "distribution is sharp" but misses "argmax drifted even though entropy stayed low" (paper §1.2). Latency-budget catches "slow" but misses "fast but wrong". |

The signals are genuinely complementary, not redundant. The advantage-margin gate is the **only** one of the four that asks "did this step help?" (improvement signal) vs "is this step done?" (sufficiency) or "is this step slow?" (budget).

### Open design question (blocks implementation)

`AdvantageMarginGate::should_recurse(pre_logits, post_logits, candidate)` consumes **policy distributions** (logits over candidate actions). The HLA reconstruction loop exposes:

- `activations: [f32; 6]` — module activations (dot-product + sigmoid projections, range ~[0, 1]). Treatable as logits over 6 "module candidates".
- `self.hla: [f32; 8]` — HLA state vector. **NOT a policy distribution** — arbitrary real values, can be negative. Log-softmax on this is mathematically valid but semantically dubious.
- `evidence.activation_entropy()` — already used by entropy-threshold.

**The cleanest integration** is to apply the gate to the 6-element `activations` vector, treating module activations as a "routing policy" over SenseKinds. `candidate` = the currently-selected top module (from `route()`). Pre = activations from step N-1, post = activations from step N.

**Risk:** the gate math expects logits-shaped inputs where `softmax(logits)` is a meaningful distribution. Module activations are already sigmoid-bounded [0, 1], not logits. Naive application may behave differently than the synthetic-recursion benchmark. Needs a targeted benchmark on real reconstruction traces before promoting.

### Tasks (GOAT-gated, behind `self_advantage_gate` feature — already default-on)

- [ ] **T5.1.1** Add `ReconstructionConfig::advantage_margin_threshold: f32` (default: `f32::NAN` = disabled; Bench 056 default 0.01 when enabled). Feature-gate behind `self_advantage_gate`.
- [ ] **T5.1.2** In `reconstruct_inner`, capture previous-step `activations` (stack buffer `[f32; 6]`), call `gate.should_recurse(&prev_activations, &activations, candidate_idx)`, halt if false.
- [ ] **T5.1.3** Benchmark: replay 1000 reconstruction traces with vs without gate. Metrics: (a) mean steps saved, (b) final-activations argmax match, (c) per-step latency overhead (<100ns target since vocab=6 is sub-µs already).
- [ ] **T5.1.4** GOAT gate: ≥1.5× steps saved at ≥99% argmax match → promote `advantage_margin_threshold` default from NaN to 0.01. Demote (or keep) entropy_threshold based on relative win.
- [ ] **T5.1.5** Document in `.docs/26_micro_belief.md` — add the gate as the 4th early-stop criterion.

**Tier:** GOAT-tier optimization (measurable improvement to existing capability, no new capability class). No plan — this issue is the tracking artifact per AGENTS.md.

---

## T5.3 — Freeze/thaw snapshot of improvement direction vector `A(·)`

**Status:** Speculative, blocked on T5.1 producing useful per-NPC `A(·)` traces.

The idea: per-NPC, the advantage-margin direction `A(·) = log π+ − log π̂` over many ticks characterizes which kinds of observations improve that NPC's predictions. Snapshotting this as a versioned latent direction vector (BLAKE3-committed `MicroRecurrentKernelSnapshot` variant — Plan 276) gives NPCs a per-personality "what improves me" fingerprint.

**Why deferred:** T5.1 isn't shipped. Without per-NPC `A(·)` traces, there's nothing to snapshot. Also: `A(·)` is per-step per-candidate, not a single direction vector — needs aggregation design (running EMA? top-K directions?).

**Re-evaluate after T5.1.3 benchmark produces real traces.**

---

## T2.2 — Wire into `LoopMode::WeightShared`

**Integration site:** `katgpt-rs/crates/katgpt-core/src/types.rs:314-324` (`LoopMode::WeightShared { loop_count }`)

### Why deferred

`WeightShared` is the transformer hot path — same layers applied `loop_count` times. Breaking early on a gate signal requires:
1. Capturing pre-recursion logits at step 0 (cheap — already computed by the forward pass).
2. Capturing post-recursion logits at each step (cheap — already computed by the readout head).
3. Inserting a `gate.should_recurse(pre, post, candidate)` check between steps.
4. Breaking the loop with the last accepted state.

Step 3 is cheap (~100ns for LLM-scale vocab, see Bench 056 G3 informational note: ~4µs at vocab=1024). The risk is touching the hot inference path and potentially regressing the no-gate case by even a few ns.

### Tasks (deferred — not blocking the shipped primitive)

- [ ] **T2.2.1** Locate the `WeightShared` loop body in the forward pass. Likely `katgpt-rs/src/loop_*` or similar (grep required).
- [ ] **T2.2.2** Add `Option<&mut AdvantageMarginGate>` parameter (None = no-op, no perf regression).
- [ ] **T2.2.3** Benchmark the no-gate path is byte-identical and ≤1ns slower than baseline.
- [ ] **T2.2.4** Benchmark the gated path on a real looped-transformer workload (use Plan 276 `LatentThoughtKernel` K-iteration as the test substrate — it's the closest shipped "weight-shared loop").

**Tier:** Deep integration, touches hot path. Defer until there's a concrete looped-transformer workload that needs it. The standalone primitive + HLA integration (T5.1) covers the immediate game-AI use case.

---

## T2.3 — `pre_recursion_logits` capture hook on `SpeculativeGenerator` trait

**Integration site:** `katgpt-rs/crates/katgpt-core/src/traits.rs:997-1007`

```rust
pub trait SpeculativeGenerator {
    type Condition;
    type Output;
    type Error;
    fn generate(&mut self, cond: &Self::Condition) -> Result<Self::Output, Self::Error>;
    fn generate_batch(&mut self, cond: &Self::Condition, n: usize) -> Result<Vec<Self::Output>, Self::Error>;
}
```

### Why deferred

Adding a `pre_recursion_logits` hook is a **trait surface change** that affects every implementor. The benefit: any `SpeculativeGenerator` could be wrapped with `AdvantageMarginGate` and get dead-compute detection for free.

The cost: every implementor needs to expose its pre-recursion logits. Some generators (e.g., game-action generators that don't have a "recursion" concept) would have to return `None` or panic.

### Recommended approach (when revisited)

Don't modify `SpeculativeGenerator`. Instead, define a **new opt-in trait**:

```rust
pub trait RecursionLogits {
    fn pre_recursion_logits(&self) -> &[f32];
    fn post_recursion_logits(&self) -> &[f32];
}
```

Generators that have a recursion loop implement `RecursionLogits`; `AdvantageMarginGate` is parametric over `P: RecursionLogits`. No trait breakage, no `Option` boilerplate on generators that don't recurse.

**Tier:** API ergonomics improvement. Defer until ≥2 recursion-capable generators exist and need unified gating. Today the gate works standalone via `should_recurse(pre, post, candidate)` — no trait change needed for the shipped use cases.

---

## Summary

| Item | Tier | Action | Status |
|------|------|--------|--------|
| T5.2 (riir-ai Super-GOAT guide) | **NOT Super-GOAT** (Q1 NO, Q2 Partial) | Verdict rendered, no guide created | ✅ Closed (this issue) |
| T5.1 (HLA evolve_hla integration) | GOAT optimization | Tracked here as T5.1.1–T5.1.5 | Open — needs benchmark |
| T5.3 (Freeze/thaw A(·) snapshot) | Speculative | Blocked on T5.1 | Open — deferred |
| T2.2 (LoopMode::WeightShared wire) | Deep integration, hot path | Tracked here as T2.2.1–T2.2.4 | Open — deferred |
| T2.3 (SpeculativeGenerator trait hook) | API ergonomics | New `RecursionLogits` trait recommended over modifying `SpeculativeGenerator` | Open — deferred |

**No Super-GOAT guide created.** No new plans created (per AGENTS.md, optimization tasks → issues). The shipped primitive (Phase 1–4) is complete and GOAT-validated; all follow-ups are integration opportunities tracked here.

---

**TL;DR:** Plan 283 Phase 1–4 is shipped and GOAT-validated (4/4 PASS, default-on). The T5.2 Super-GOAT re-evaluation is complete: **NOT Super-GOAT** (Q1 prior art in `evolve_hla` + 3 existing HLA early-stop criteria; Q2 is optimization not new capability). T5.1 (HLA integration) is a real GOAT-tier optimization, complementary to existing `max_steps` + `entropy_threshold` + `adaptive_budget` criteria — but needs a targeted benchmark on real reconstruction traces because module activations aren't true logits. T5.3 blocked on T5.1. T2.2/T2.3 are deferred deep integrations with recommended non-breaking approaches documented above. No guide, no new plans — this issue is the tracking artifact.
