# Plan 304: Gain/Cost Loop Halting Primitive

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/282_LoopCoder_V2_Gain_Cost_Loop_Halting.md](../.research/282_LoopCoder_V2_Gain_Cost_Loop_Halting.md)
**Private guide:** [riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md](../../../riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md)
**Source paper:** [arxiv 2606.18023](https://arxiv.org/abs/2606.18023) — LoopCoder-v2 (Yang et al., 2026)
**Target:** `crates/katgpt-core/src/gain_cost_halt.rs` (new module) + Cargo feature `gain_cost_halt`
**Status:** Active — Phase 1 complete (kernel + 24/24 G1 mechanics tests shipped), Phase 2 (forward_looped wiring) + Phase 3 (docs) deferred.

---

## Goal

Ship a substrate-agnostic `GainCostLoopHalter` that decides — per loop, per dispatch, at runtime — whether to continue looping or halt, based on the LoopCoder-v2 "gain/cost scissors" criterion: **halt when marginal refinement gain < marginal drift cost × τ**. This is the open primitive for the Super-GOAT identified in Research 282; the private selling-point wiring (per-NPC reasoning depth) lands in riir-ai per the guide at `.research/149`.

The primitive composes with the shipped elastic loop override (`Config::effective_loop_count`, Issue 035) — instead of the caller passing a static `Some(L)`, the caller passes the halter's decision each loop. It reuses the shipped `effective_rank` from Plan 152 (River-Valley Diagnostics) as the primary gain signal, and is designed to compose with coherence-decay (latent_functor) and staleness (HLA) cost signals wired in riir-ai.

**GOAT gate:** feature flag `gain_cost_halt`, opt-in. Must pass G1–G5 (see Research 149 §5) before any default promotion. Demote if G2 (≥75% crowd-NPC compute savings) or G3 (no important-NPC regression) fail.

---

## Phase 1 — Core Kernel (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/gain_cost_halt.rs` behind `gain_cost_halt` feature. Re-export from `crates/katgpt-core/src/lib.rs`.

- [x] **T1.2** Define the halter state struct (zero-alloc, hot-path-safe):

```rust
/// Per-loop state for the gain/cost halting criterion (Research 282 / Plan 304).
///
/// Tracks the signals needed to decide whether to continue looping: the
/// previous loop's effective rank (for the gain curve), the previous loop's
/// hidden-state snapshot (for step-size + angular-change computation), and
/// the oscillation counter (for early halt on cos θ < 0).
///
/// State size: 12 bytes + 1 `Option<&[f32]>` borrow (no allocation). The
/// hidden-state borrow is valid only within a single `forward_looped()` call;
/// the halter does not own a copy.
#[derive(Clone, Debug, Default)]
pub struct GainCostLoopHalter {
    /// Effective rank at the previous loop (for delta computation).
    /// None on the first loop (no previous to compare against).
    prev_erank: Option<f32>,
    /// Step size at the previous loop (||h^(r) - h^(r-1)||₂). Used for the
    /// angular-change cos θ computation.
    prev_step: f32,
    /// Count of consecutive loops where cos θ < 0 (oscillation detector).
    /// Halts when this reaches `oscillation_patience`.
    oscillation_count: u8,
    /// Config: halt when gain < cost × tau. Default tau = 1.0.
    tau: f32,
    /// Config: halt after this many consecutive oscillatory loops. Default 1.
    oscillation_patience: u8,
    /// Config: L_min floor (refuse to halt below this loop index). Default 1.
    l_min: u8,
}
```

- [x] **T1.3** Implement `halt_decision(&mut self, loop_idx: usize, gain: f32, cost: f32, cos_theta: f32) -> HaltDecision`:

```rust
/// Result of a single gain/cost halt evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaltDecision {
    /// Continue looping — gain >= cost × tau and no oscillation detected.
    Continue,
    /// Halt now — gain < cost × tau, OR oscillation count reached patience.
    /// The caller should exit the loop and use the current hidden state.
    Halt { reason: HaltReason },
    /// Refused — loop_idx < l_min. Continue regardless of gain/cost.
    /// Protects representational capacity (ELT §1.4: sub-floor loops collapse).
    RefusedFloor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaltReason {
    /// Marginal refinement gain dropped below drift cost × tau.
    GainBelowCost,
    /// Update direction reversed (cos θ < 0) for `oscillation_patience` loops.
    Oscillation,
}

impl GainCostLoopHalter {
    pub fn halt_decision(
        &mut self,
        loop_idx: usize,   // 1-based: the loop just completed
        gain: f32,         // e.g. effective-rank delta, or coherence improvement
        cost: f32,         // e.g. coherence decay, or staleness
        cos_theta: f32,    // alignment of last two update directions ([-1, 1])
    ) -> HaltDecision {
        // L_min floor — refuse to halt below representational minimum.
        if loop_idx < self.l_min as usize {
            return HaltDecision::RefusedFloor;
        }
        // Oscillation early-halt — catches what stability-only primitives miss.
        if cos_theta < 0.0 {
            self.oscillation_count = self.oscillation_count.saturating_add(1);
            if self.oscillation_count >= self.oscillation_patience {
                return HaltDecision::Halt { reason: HaltReason::Oscillation };
            }
        } else {
            self.oscillation_count = 0;
        }
        // Gain/cost scissors — the primary criterion.
        if gain < cost * self.tau {
            return HaltDecision::Halt { reason: HaltReason::GainBelowCost };
        }
        HaltDecision::Continue
    }
}
```

- [x] **T1.4** Implement signal extractors (zero-alloc, accept `&mut scratch` buffers): — **DEVIATION:** `crate::data_probe::geometry::effective_rank` lives in the ROOT crate (`katgpt-rs/src/data_probe/`), not katgpt-core; katgpt-core cannot depend on the root (circular dep). `hidden_erank` ships a local self-contained Roy & Vetterli entropy-of-eigenvalue-spectrum implementation (column-mean centering → min(S,d)×min(S,d) Gram → in-place Jacobi eigenvalues → normalize → Shannon entropy → exp(entropy)).

```rust
/// Compute effective rank from a hidden-state matrix (S × d), reusing the
/// River-Valley Diagnostics kernel from Plan 152.
///
/// Caller passes the hidden state as a flat `&[f32]` of shape `[S, d]` (row-major),
/// plus a scratch buffer for singular values. The function delegates to
/// `data_probe::geometry::effective_rank` (already shipped, default-on).
#[inline]
pub fn hidden_erank(hidden: &[f32], s: usize, d: usize, scratch_sv: &mut [f32]) -> f32 {
    // Delegate to the shipped River-Valley Diagnostics kernel.
    crate::data_probe::geometry::effective_rank(hidden, s, d, scratch_sv)
}

/// Compute step size δ = ||h^(r) - h^(r-1)||₂ between two consecutive loops.
/// Zero-allocation: caller passes both hidden states as `&[f32]`.
#[inline]
pub fn step_size(h_curr: &[f32], h_prev: &[f32]) -> f32 {
    debug_assert_eq!(h_curr.len(), h_prev.len());
    let mut sum_sq = 0.0f32;
    for (a, b) in h_curr.iter().zip(h_prev.iter()) {
        let diff = a - b;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Compute angular change cos θ between two successive update directions.
/// curr_step = h^(r) - h^(r-1); prev_step = h^(r-1) - h^(r-2).
/// Returns [-1, 1]: 1 = same direction (convergent), <0 = reversal (oscillatory).
#[inline]
pub fn angular_change(curr_step: &[f32], prev_step: &[f32]) -> f32 {
    debug_assert_eq!(curr_step.len(), prev_step.len());
    let mut dot = 0.0f32;
    let mut norm_curr = 0.0f32;
    let mut norm_prev = 0.0f32;
    for (a, b) in curr_step.iter().zip(prev_step.iter()) {
        dot += a * b;
        norm_curr += a * a;
        norm_prev += b * b;
    }
    let denom = (norm_curr * norm_prev).sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}
```

- [x] **T1.5** G1 mechanics unit tests:
  - `halt_decision_gain_below_cost_halt`
  - `halt_decision_gain_above_cost_continue`
  - `halt_decision_refused_below_l_min`
  - `halt_decision_oscillation_after_patience`
  - `halt_decision_oscillation_resets_on_positive_cos`
  - `step_size_zero_for_identical_states`
  - `angular_change_zero_for_zero_step`
  - `angular_change_negative_for_reversal`
  - No NaN/Inf in any path; `HaltDecision` enum is `#[repr(u8)]`-compatible for cache-line friendliness. — **24/24 PASS** (added NaN-safety + tau-scaling + tall-matrix + rank-one + collapsed + scratch-contract + orthogonal + aligned tests beyond the plan's minimum list).

- [x] **T1.6** Add `gain_cost_halt` feature to `crates/katgpt-core/Cargo.toml`. Default = OFF (opt-in until G1–G5 pass).

---

## Phase 2 — LT2 Forward Path Wiring

### Tasks

- [ ] **T2.1** Extend `forward_looped()` to accept a halter instance. The signature gains an optional `halter: Option<&mut GainCostLoopHalter>` parameter. When `Some`, the loop evaluates `halt_decision()` after each iteration and exits early on `Halt`. When `None`, behavior is byte-identical to pre-Plan-304.

- [ ] **T2.2** The halter composes with the existing `elastic_loop_override` (Issue 035): if the caller passes `Some(L)`, the halter is ignored (static override wins). If the caller passes `None` AND provides a halter, the halter decides L dynamically. This keeps the API backward-compatible.

- [ ] **T2.3** Per-loop signal extraction inside the loop body:
  - After loop r completes, compute `erank(h(r))` via `hidden_erank` using the shipped River-Valley Diagnostics kernel. The gain signal is `erank(h(r)) - erank(h(r-1))` (positive = enriching, negative = narrowing).
  - Compute `step_size(h(r), h(r-1))` for the angular-change signal.
  - The cost signal is configurable: default is a fixed `cost_floor` (e.g., 0.01 × the loop-1 step size — a fixed tax analogous to LoopCoder-v2's flat Ω(r)); riir-ai can override with coherence-decay or staleness.
  - Call `halter.halt_decision(r, gain, cost, cos_theta)`. On `Halt`, break.

- [ ] **T2.4** G2 crowd-NPC synthetic benchmark: construct a synthetic bimodal loop suite (easy inputs: gain drops below cost after loop 1; hard inputs: gain stays above cost to L_max). Verify the halter halts easy inputs early (≥75% compute saved) and runs hard inputs to completion.

- [ ] **T2.5** G3 no-regression benchmark: verify hard-input outputs with the halter are bit-identical to fixed-L_max outputs (the halter only halts early when it's safe to do so).

---

## Phase 3 — Documentation & Feature Gate

### Tasks

- [ ] **T3.1** Update `.docs/02_architecture.md` with a new "Gain/Cost Loop Halting" subsection under the LT2 Looped Forward Pass section.
- [ ] **T3.2** Update `.docs/15_paper_feature_comparison.md` with Research 282 / Plan 304 row.
- [ ] **T3.3** Update `README.md` Feature Showcase with a Gain/Cost Loop Halting entry.
- [ ] **T3.4** Add an example `examples/gain_cost_halt_demo.rs` showing the halter on a synthetic looped forward pass (gain curve collapses, halter halts at the crossover).
- [ ] **T3.5** G5 feature isolation test: `cargo check --no-default-features`, `cargo hack check --each-feature`, `cargo check --all-features` all pass. Zero overhead when `gain_cost_halt` is off (the `Option<&mut GainCostLoopHalter>` is `None` and the branch is predicted-not-taken).

---

## Open Questions

1. **Should the cost signal default to a fixed tax or a per-loop computed value?** LoopCoder-v2's Ω(r) is empirically flat (a fixed tax). Our default `cost_floor` mirrors this. But for HLA evolution, the natural cost is staleness (which grows over ticks, not loops). Resolution: make the cost a closure or trait object — default is fixed tax; riir-ai overrides with staleness.

2. **Should the halter's gain signal be effective-rank delta or output shift Δp?** Effective rank is cheaper (no output head evaluation) and is the paper's recommended lightweight diagnostic. Output shift is more direct but requires applying the LM head each loop. Default: effective-rank delta; output shift as opt-in via config.

3. **Does the halter need to be deterministic across nodes for sync/replay?** Yes — the gain/cost are pure functions of deterministic hidden states, so L is deterministic. But we need a G1 test that verifies this: same input on two nodes produces the same halt count. (Documented in Research 149 §5 G1.)

4. **How does the halter interact with `LoopMode::TrainingFree`?** Training-Free Loop (Plan 136) uses damped sub-stepping which is itself a stability mechanism. The halter should be compatible — it evaluates the gain/cost after each K-stage RK sub-step, same as for WeightShared. But the semantics differ (TF-Loop targets a fixed endpoint t=1, so halting mid-sub-step means accepting a less-refined endpoint). Resolution: TF-Loop halting is Phase 2.5 (defer until LT2 WeightShared wiring is proven).

---

## Risks

| Risk | Mitigation |
|------|------------|
| Effective-rank computation per loop is too expensive | Compute on a subsample of tokens (e.g., 32 random rows of the S×d matrix); or cache the SVD from the previous loop and update incrementally |
| Halter halts too aggressively on small models (low representational diversity) | L_min floor protects; tune `tau` per model size |
| Halter never halts on well-trained models (gain always > cost) | This is correct behavior — run to L_max. The savings come from the crowd-NPC tier where gain collapses fast. |
| Feature flag interaction with `lt2_looped` and `tf_loop` | Document the dependency graph: `gain_cost_halt` requires `lt2_looped` (Phase 2) or `tf_loop` (Phase 2.5). Phase 1 (the kernel) has no deps. |

---

## TL;DR

Plan 304 ships the open primitive for Research 282 (LoopCoder-v2 Super-GOAT): a `GainCostLoopHalter` kernel that decides per-loop whether to continue, based on the gain/cost scissors criterion (halt when marginal refinement < marginal drift × τ). Phase 1 ships the substrate-agnostic kernel (struct + `halt_decision` + signal extractors that reuse Plan 152's `effective_rank`). Phase 2 wires it into `forward_looped()` via the existing `elastic_loop_override` path (Issue 035) — backward-compatible: `None` halter = current behavior, `Some(halter)` = gain/cost-gated. The private selling-point wiring (per-NPC reasoning depth, HLA belief evolution, latent_functor coherence) lands in riir-ai per Research 149. GOAT gate G1–G5 (Research 149 §5) must pass before default promotion; G2 (≥75% crowd-NPC compute savings) and G4 (oscillation detection catches what stability misses) are the headline gates. Latent-vs-raw: gain/cost signals are local latent; halt count L is a deterministic raw scalar safe to sync/replay.
