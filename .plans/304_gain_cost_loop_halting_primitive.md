# Plan 304: Gain/Cost Loop Halting Primitive

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/282_LoopCoder_V2_Gain_Cost_Loop_Halting.md](../.research/282_LoopCoder_V2_Gain_Cost_Loop_Halting.md)
**Private guide:** [riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md](../../../riir-ai/.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md)
**Source paper:** [arxiv 2606.18023](https://arxiv.org/abs/2606.18023) — LoopCoder-v2 (Yang et al., 2026)
**Target:** `crates/katgpt-core/src/gain_cost_halt.rs` (new module) + Cargo feature `gain_cost_halt`
**Status:** Active — Phase 1 complete (kernel + 24/24 G1 mechanics tests shipped). Phase 2 complete (forward_looped wiring: T2.1–T2.3 done; T2.4/T2.5 deferred — requires synthetic loop suite harness). Phase 3 done (T3.1 architecture doc, T3.3 README entry, T3.4 demo example, T3.5 feature isolation PASS; T3.2 skipped — comparison doc out of scope for Research 282). 27/27 kernel tests + 28/28 forward_looped integration tests PASS.

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

- [x] **T2.1** Extend `forward_looped()` to accept a halter instance. The signature gains an optional `halter: Option<&mut GainCostLoopHalter>` parameter (feature-gated `gain_cost_halt`, last param after `elastic_loop_override`). When `Some`, the loop evaluates `halt_decision()` after each iteration and exits early on `Halt`. When `None`, behavior is byte-identical to pre-Plan-304. **All 9 call sites updated** (bench_108_lt2_looped ×6, goat_108_lt2_looped ×1, issue_035_any_time_lt2_dispatch ×1, t2_2_weight_shared_gate ×1) with `#[cfg(feature = "gain_cost_halt")] None,` mirroring the recursion_gate pattern. **NECESSARY DEVIATION:** added `gain_cost_halt = ["katgpt-core/gain_cost_halt"]` feature alias to root `Cargo.toml` — the feature only existed in katgpt-core (Phase 1), not re-exposed in the root crate, so `#[cfg(feature = "gain_cost_halt")]` in transformer.rs could never activate. One feature line added (minimal, required for T2.1 to compile).

- [x] **T2.2** The halter composes with the existing `elastic_loop_override` (Issue 035): if the caller passes `Some(L)`, the halter is ignored (static override wins, `halter_active = elastic_loop_override.is_none()`). If the caller passes `None` AND provides a halter, the halter decides L dynamically. API backward-compatible. Tested via `issue_035_any_time_lt2_dispatch` (13/13 PASS) and `goat_108_lt2_looped` (11/11 PASS).

- [x] **T2.3** Per-loop signal extraction inside the loop body:
  - **DEVIATION:** gain signal = `step_size(ctx.x, ctx.prev_h)` = `||h^(tau) - h^(tau-1)||_2`, NOT effective-rank delta. The per-loop hidden state in `forward_looped` is a SINGLE vector `ctx.x[..n]` (S=1), for which `hidden_erank` returns 0.0 (degenerate — the kernel short-circuits on `s == 1`). `step_size` is monotone in refinement, cheaper than erank, and the kernel ships it for exactly this use. See Open Question 2 resolution.
  - `cos_theta` = `angular_change(curr_step, prev_step)` between successive update directions. `prev_step_buf` / `curr_step_buf` allocated ONCE per `forward_looped` call (not per iteration) — honors the hot-loop rule.
  - cost = fixed tax (`cost_floor`), cached as `0.01 × first_loop_step_size` on `tau == 1` — mirrors LoopCoder-v2's flat Ω(r). riir-ai can override with coherence-decay/staleness.
  - Call `halter.halt_decision(tau + 1, gain, cost, cos_theta)`. On `Halt`, break.
  - **Added public setters** `update_prev_step` / `update_prev_erank` / `prev_step` getter on `GainCostLoopHalter` — fields are `pub(crate)` and the wiring lives in the ROOT crate (not katgpt-core), so the wiring cannot access them directly.
  - **Added 3 G1 wiring tests** (`update_prev_step_setter_round_trips`, `update_prev_erank_setter_round_trips`, `refused_floor_never_halts_when_l_min_above_loop_count`) — 27/27 kernel tests PASS.

- [ ] **T2.4** G2 crowd-NPC synthetic benchmark — **deferred: requires synthetic loop suite harness.**

- [ ] **T2.5** G3 no-regression benchmark — **deferred: requires synthetic loop suite harness.**

---

## Phase 3 — Documentation & Feature Gate

### Tasks

- [x] **T3.1** Added "Gain/Cost Loop Halting" subsection to `.docs/02_architecture.md` under the LT2 Looped Forward Pass section. Covers the scissors criterion, composition with `elastic_loop_override` (static wins) + `recursion_gate` (both fire independently), signal extractors (step_size gain — DEVIATION noted; cost_floor default; cos_theta for oscillation), the RefusedFloor → Continue → Halt state machine, and all 4 open-question resolutions.

- [ ] **T3.2** Update `.docs/15_paper_feature_comparison.md` with Research 282 / Plan 304 row — **skipped:** the comparison doc's scope is Research papers 00–069; Research 282 (LoopCoder-v2) is outside that range. The architecture doc (T3.1) + README entry (T3.3) cover the mapping instead.

- [x] **T3.3** Added a one-line entry for `gain_cost_halt` to `README.md`'s "🔀 Opt-In & Gated Features" table. Notes Plan 304/Research 282/arXiv:2606.18023, the gain=step_size deviation, and G2/G3 deferred status.

- [x] **T3.4** Created `examples/gain_cost_halt_demo.rs` — pure-kernel synthetic demo. Simulates 10 loops with geometrically-decaying step_size (crowd-NPC regime), fixed `cost_floor=0.1`, prints a table, and shows the halter firing at the gain/cost crossover (loop 5, saving 50% compute). Also validates the oscillation path. Runs: `cargo run --example gain_cost_halt_demo --features gain_cost_halt`.

- [x] **T3.5** G5 feature isolation:
  - `cargo check --no-default-features` — PASS (only pre-existing warnings).
  - `cargo check --features gain_cost_halt` — PASS.
  - `cargo check --all-features` — 10 pre-existing errors in unrelated modules (`percepta/evaluator.rs`, `proof_cert/wasm_proof_witness.rs`, `feedback_bandit.rs`, `linoss.rs`, etc. — other agents' in-progress work); NONE in gain_cost_halt or forward_looped files.
  - `cargo check --features "gain_cost_halt,lt2_looped,sleep_consolidation,weight_shared_advantage_gate"` — PASS (full composition).
  - `cargo-hack` installed (v0.6.45); `--each-feature` not run to completion because it triggers `--all-features` combos that hit the same pre-existing unrelated failures. Scoped powerset on the 4 relevant features PASS.
  - The `Option<&mut GainCostLoopHalter>` param is zero-cost when feature is off (cfg-stripped from signature) and predicted-not-taken when feature is on but halter is `None`.

---

## Open Questions (resolutions after Phase 2)

1. **Should the cost signal default to a fixed tax or a per-loop computed value?** ✅ **Resolved:** Phase 2 ships the fixed-tax default (`cost_floor = 0.01 × first_loop_step_size`, cached on `tau == 1`) — mirrors LoopCoder-v2's flat Ω(r). riir-ai can override with coherence-decay/staleness by not using the Phase-2 code path (e.g., computing its own cost signal and calling `halt_decision` directly). The kernel's `halt_decision` accepts any `cost: f32`, so the override is a caller-side choice.

2. **Should the halter's gain signal be effective-rank delta or output shift Δp?** ✅ **Resolved (with DEVIATION):** Neither, as it turns out. The per-loop hidden state in `forward_looped` is a SINGLE vector `ctx.x[..n]` (one row, S=1), for which `hidden_erank` returns 0.0 (degenerate — the kernel short-circuits on `s == 1` because a single row has no variance to measure). **Phase 2 uses `step_size` as the gain signal:** `||h^(tau) - h^(tau-1)||_2`. This is monotone in refinement (the hidden state travels less each loop as it converges), cheaper than erank (no Gram matrix / Jacobi sweep), and the kernel ships `step_size` for exactly this use. If a future variant stores a multi-row hidden-state matrix per loop, erank-delta can be swapped in via `hidden_erank` — the kernel already supports it. Output-shift Δp is still deferred (requires applying the LM head each loop — the `recursion_gate` path already does this for logit-improvement; combining the two is future work).

3. **Does the halter need to be deterministic across nodes for sync/replay?** ✅ **Resolved + tested:** Yes. gain/cost/cos_theta are pure functions of deterministic hidden state, so L is deterministic. Tested by `refused_floor_never_halts_when_l_min_above_loop_count` (G1 determinism guarantee: when the halter is configured to never halt via `l_min=255`, it is a pure no-op and `forward_looped` output is bit-identical to the no-halter path). The integration tests (`issue_035_any_time_lt2_dispatch` `none_path_is_byte_identical_across_runs`, `goat_108_lt2_looped`) confirm the `halter=None` path is byte-identical across runs.

4. **How does the halter interact with `LoopMode::TrainingFree`?** ✅ **Resolved (deferred):** TF-Loop halting is Phase 2.5, DEFERRED. Phase 2 only wires into `forward_looped` (WeightShared path), not `forward_tf_looped`. The semantics differ (TF-Loop targets a fixed endpoint t=1, so halting mid-sub-step means accepting a less-refined endpoint) and the wiring needs separate validation.

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

Plan 304 ships the open primitive for Research 282 (LoopCoder-v2 Super-GOAT): a `GainCostLoopHalter` kernel that decides per-loop whether to continue, based on the gain/cost scissors criterion (halt when marginal refinement < marginal drift × τ). Phase 1 shipped the substrate-agnostic kernel (struct + `halt_decision` + signal extractors). **Phase 2 (this increment) wires it into `forward_looped()`** via a new feature-gated `halter: Option<&mut GainCostLoopHalter>` parameter — backward-compatible: `None` halter = current behavior (byte-identical), `Some(halter)` = gain/cost-gated. **Key deviation:** the gain signal is `step_size` (`||h^(tau) - h^(tau-1)||_2`), NOT effective-rank delta as planned — the per-loop hidden state is a single vector (S=1), making erank degenerate. The halter composes with `elastic_loop_override` (static wins, T2.2) and `recursion_gate` (both fire independently). Phase 3 added the architecture doc subsection, README entry, and a pure-kernel demo example (`gain_cost_halt_demo`) that shows the halter firing at the gain/cost crossover (50% compute saved on geometrically-decaying inputs). **Validation:** 27/27 kernel tests + 28/28 forward_looped integration tests PASS; feature isolation confirmed (no leakage in `--no-default-features`, clean compile in `--features gain_cost_halt` and the full composition combo). G2 (≥75% crowd-NPC compute savings) and G3 (no important-NPC regression) are DEFERRED — they require a synthetic loop suite harness that doesn't exist yet. The private selling-point wiring (per-NPC reasoning depth, HLA belief evolution, latent_functor coherence) lands in riir-ai per Research 149. Latent-vs-raw: gain/cost signals are local latent; halt count L is a deterministic raw scalar safe to sync/replay.
