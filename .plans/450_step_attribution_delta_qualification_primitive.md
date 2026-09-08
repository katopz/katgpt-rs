# Plan 450: Step-Attribution Δ-Qualification Primitive

**Date:** 2026-07-06
**Research:** [katgpt-rs/.research/381_SkillAdaptor_Step_Level_Fault_Attribution_Delta_Qualification.md](../.research/381_SkillAdaptor_Step_Level_Fault_Attribution_Delta_Qualification.md)
**Private guide:** [riir-ai/.research/313_Step_Level_Fault_Attribution_Commit_Gate_Guide.md](../../riir-ai/.research/313_Step_Level_Fault_Attribution_Commit_Gate_Guide.md)
**Runtime wiring:** [riir-ai/.plans/569_step_attribution_branch_wiring.md](../../riir-ai/.plans/569_step_attribution_branch_wiring.md) (renumbered from 313)
**Source paper:** [arXiv:2606.01311](https://arxiv.org/abs/2606.01311) — SkillAdaptor, Yu et al. 2026
**Target:** `crates/katgpt-pruners/src/step_attribution_qualifier.rs` (new module) + Cargo feature `step_attribution_qualifier`
**Status:** ✅ COMPLETE (Phases 1-5 done; Phase 4 T4.1 done 2026-07-06; T4.2 deferred per plan)

---

## Goal

Ship a generic modelless primitive that takes a failed trajectory, a candidate state `K+`, a baseline state `K`, and a replay window, and returns an accept/reject verdict based on a Δ-based re-execution comparison (SkillAdaptor eq. 8). This is the explicit "did this update actually help?" gate that TrajectoryDoctor (Plan 223, ships `localize_failure`) and the existing WasmTestGate Δ-field do not unify. Sibling to TrajectoryDoctor in `katgpt-pruners`. Opt-in feature flag; **promote to default ONLY after the riir-ai Plan 313 G6 quality-parity PoC passes** (per §3.6 defend-wrong rule).

The primitive is generic (no game semantics): it works on any candidate/baseline pair where (a) the candidate is a deterministic function of the baseline + a proposed mutation, and (b) the trajectory can be re-executed deterministically given the candidate/baseline state.

## Stack slot (per the per-stack promote/demote discipline)

This primitive does NOT fit a transformer stack slot (attention/KV/sampling/speculative/pruning). It sits in the **cognitive-branch / skill-lifecycle stack** alongside `TrajectoryDoctor`, `FailureEpisodeStore`, `ClosedUnitCompactionGate`. Promotion/demotion is tracked against the shipped `TrajectoryDoctor` family — if `StepAttributionQualifier` wins the variance-reduction comparison in G6, it becomes the recommended commit gate for cognitive-branch / skill-lifecycle consumers; `TrajectoryDoctor` alone remains for localization-only use cases.

## Phase 1 — Skeleton + Trait (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-pruners/src/step_attribution_qualifier.rs` behind feature `step_attribution_qualifier`. Module doc references Plan 381 + Research 381 + SkillAdaptor arXiv:2606.01311 eq. 8.
- [x] **T1.2** Define the core trait + types (generic over state `K`, replay input `I`, score `S`):

```rust
//! Step-Attribution Δ-Qualification Primitive (Plan 381, Research 381)
//!
//! Generic modelless instantiation of SkillAdaptor's eq. 8:
//!   Δ = E_q[M(q; K+)] - E_q[M(q; K)]
//! Commit candidate state K+ iff Δ >= 0.
//!
//! Sibling to [`crate::trajectory_doctor::TrajectoryDoctor`] (Plan 223).
//! TrajectoryDoctor localizes the fault; this gate decides whether to commit
//! the proposed fix.

use crate::trajectory_doctor::FailureSite;

/// A proposed mutation to a baseline state, producing a candidate state.
pub trait CandidateMutation<K> {
    /// Apply this mutation to the baseline, producing the candidate state K+.
    fn apply_to(&self, baseline: &K) -> K;
}

/// Deterministic re-execution of a replay window under a given state.
/// Implementors guarantee bit-identical results for the same (state, inputs) pair.
pub trait ReplayExecutor<K, I, S> {
    /// Re-execute the replay window `inputs` under state `k`, returning per-step scores.
    fn replay(&self, k: &K, inputs: &[I]) -> Vec<S>;
}

/// Aggregate a per-step score slice into a single comparable metric.
/// Typically `Sum` for CLR reward; may be `Mean` for normalized metrics.
pub trait ScoreAggregator<S> {
    fn aggregate(&self, scores: &[S]) -> S;
}

/// The Δ≥0 commit gate. Wraps a ReplayExecutor + ScoreAggregator.
pub struct StepAttributionQualifier<K, I, S, E, A> {
    executor: E,
    aggregator: A,
    /// Acceptance threshold (default 0.0 = SkillAdaptor's Δ ≥ 0).
    /// Positive values encode a "strictly better" requirement.
    pub threshold: S,
    _marker: core::marker::PhantomData<(K, I)>,
}

/// The verdict returned by qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationVerdict {
    /// Δ ≥ threshold — commit the candidate.
    Commit { delta_above_threshold: bool },
    /// Δ < threshold — rollback to baseline.
    Rollback { delta_below_threshold: bool },
}

impl<K, I, S, E, A> StepAttributionQualifier<K, I, S, E, A>
where
    E: ReplayExecutor<K, I, S>,
    A: ScoreAggregator<S>,
    S: PartialOrd + Copy + core::ops::Sub<Output = S>,
{
    /// Run the full Δ≥0 qualification on a candidate mutation.
    ///
    /// 1. Apply mutation to baseline → K+
    /// 2. Replay window under K → baseline_scores
    /// 3. Replay window under K+ → candidate_scores
    /// 4. Aggregate both
    /// 5. Δ = aggregate(K+) - aggregate(K); Commit iff Δ ≥ threshold
    pub fn qualify(
        &self,
        baseline: &K,
        mutation: &dyn CandidateMutation<K>,
        replay_inputs: &[I],
    ) -> QualificationVerdict {
        let candidate = mutation.apply_to(baseline);
        let baseline_scores = self.executor.replay(baseline, replay_inputs);
        let candidate_scores = self.executor.replay(&candidate, replay_inputs);
        let delta = self.aggregator.aggregate(&candidate_scores)
            - self.aggregator.aggregate(&baseline_scores);
        if delta >= self.threshold {
            QualificationVerdict::Commit { delta_above_threshold: true }
        } else {
            QualificationVerdict::Rollback { delta_below_threshold: true }
        }
    }
}
```

- [x] **T1.3** Define the **tick-attribution** extension trait (the SkillAdaptor Localize+Link fused into one modelless call, generalizing `TrajectoryDoctor::localize_failure` from token-index to a generic tick-index):

```rust
/// A localized fault: which tick, which "responsible direction", improvement hint.
/// Generalizes [`crate::trajectory_doctor::FailureSite`] from token-index to tick-index
/// + adds the SkillAdaptor Linker's responsibility weight.
pub struct TickFaultSite<Dir, W> {
    /// The first actionable fault tick in the replay window.
    pub tick_idx: usize,
    /// The violated predicate / observation (mirrors FailureSite::violated_predicate).
    pub violated: String,
    /// Per-direction responsibility weights (SkillAdaptor eq. 6 output).
    /// `weights[j] = sigmoid(dot(hla_delta_at_t_star, direction_j))`.
    pub responsibility: Vec<W>,
    /// Argmax direction index — the "responsible skill/branch".
    pub responsible_idx: usize,
    /// Marker for the generic direction type (e.g. HLA direction vector).
    pub _marker: core::marker::PhantomData<Dir>,
}

/// SkillAdaptor's Localize (eq. 5) + Link (eq. 6) fused into one modelless call.
/// Implementors find the first fault tick in a trajectory, then project the
/// state-delta at that tick onto a direction set to attribute responsibility.
pub trait StepLocalizer<Dir, W> {
    /// Given a trajectory of per-tick state-deltas + CLR scores + a direction set,
    /// return the first actionable fault + responsibility weights.
    fn localize_and_link(
        &self,
        trajectory_deltas: &[Dir],
        trajectory_scores: &[W],
        directions: &[Dir],
        tau_reliable: W,
    ) -> Option<TickFaultSite<Dir, W>>
    where
        Dir: AsRef<[W]>,
        W: Copy + PartialOrd + core::ops::Sub<Output = W>;
}
```

- [x] **T1.4** Wire the feature flag into `crates/katgpt-pruners/Cargo.toml` + `lib.rs` module re-export. Verify `cargo check -p katgpt-pruners --features step_attribution_qualifier` is clean.
- [x] **T1.5** Update root `Cargo.toml` passthrough feature (mirror the `attention_matching` / `closed_unit_compaction` pattern). Verify `cargo check --features step_attribution_qualifier` is clean.

## Phase 2 — Tests + Doc Example

### Tasks

- [x] **T2.1** Unit tests for the Δ≥0 gate: candidate beats baseline → `Commit`; candidate equals baseline → `Commit` (Δ=0 ≥ 0); candidate worse → `Rollback`. Use a deterministic toy `ReplayExecutor` (e.g. `SumExecutor` over `&[f32]`).
- [x] **T2.2** Unit tests for the threshold variant: with `threshold = 0.1`, candidate-with-Δ=0.05 → `Rollback` (strictly-better requirement).
- [x] **T2.3** Unit tests for `StepLocalizer`: synthetic trajectory with a known fault tick + responsibility weights; assert `localize_and_link` returns the correct `tick_idx` + `responsible_idx`.
- [x] **T2.4** Doc-test example in the module doc showing the canonical usage: `let verdict = qualifier.qualify(&baseline_k, &mutation, &replay_window); match verdict { Commit => commit(), Rollback => rollback() }`.
- [x] **T2.5** Run `cargo test -p katgpt-pruners --features step_attribution_qualifier --lib` — all green. (14/14 tests pass, 2026-07-06.)

## Phase 3 — Latency Bench (G4)

### Tasks

- [x] **T3.1** Add a criterion bench `crates/katgpt-pruners/src/step_attribution_qualifier.rs` measuring `qualify()` latency for W=16/32/64/128 replay windows with a no-op `ReplayExecutor` (isolates gate overhead from executor overhead).
  - **DEV NOTE:** bench lives at `benches/step_attribution_qualifier_bench.rs` (root benches/, not crate-local) to match the existing convention (`salience_tri_gate_bench.rs`, `cucg_bench.rs`). Uses `std::time::Instant` + `harness = false`, not Criterion — DRY with the existing crate convention.
- [x] **T3.2** Document the sub-ms gate-overhead target (excluding executor). Gate overhead = aggregate + compare + branch; should be < 1µs for W=64.
  - **RESULT:** 13 ns aggregate-only at W=64 (76.9× margin under 1 µs target). End-to-end `qualify()` 119 ns at W=64.
- [x] **T3.3** Record results in `katgpt-rs/.benchmarks/381_step_attribution_qualifier_goat.md` (defer creation until bench runs).

## Phase 4 — ClosedUnitCompactionGate + WasmTestGate Adapter Shims

### Tasks

- [x] **T4.1** Ship an adapter impl showing `StepAttributionQualifier` subsumes the `WasmTestGate::avg_reward_delta` pattern from R172 ITSE — a `WasmTestGateAdapter` that wraps the existing `WasmTestGate` field as a `StepAttributionQualifier` with `threshold = 0.0`. Document in module doc that R172's gate is the prior art for new-pruner registration; this primitive generalizes to all candidate/baseline pairs.
  - **DEV NOTE (honesty correction):** Research 381 §2.3 overclaimed R172's `WasmTestGate { avg_reward_delta }` as "SHIPPED". It was **proposed** in R172 §3 "Mechanism 3" but the shipped `WasmTestGate` (`skill_test.rs`) simplified to coverage-only (no Δ field). The adapter restores the R172-proposed Δ acceptance pattern as a special case of `StepAttributionQualifier`.
  - **Implementation:** `WasmTestGateAdapter` + `ScalarStateExecutor` + `ReplaceScalar` added to `step_attribution_qualifier.rs`. Delegates to `StepAttributionQualifier::new(ScalarStateExecutor, SumAggregator, 0.0)` — proves subsumption by construction.
  - **Module doc updated:** added "# Prior art: R172 ITSE `WasmTestGate`" section.
  - **Tests:** 6 new tests (commit/rollback/equal/strict-threshold/subsumption-sweep/default-equals-new). All 20 tests green (14 original + 6 new). Doc-test still passes. `--all-features` clean (G6 no-regression).
- [-] **T4.2** **DEFERRED** — adapter for `ClosedUnitCompactionGate` (Plan 333). CUCG's rubric is structurally different (predicate-list + FireRule, not Δ-comparison); the adapter would translate a rubric verdict into a ScoreAggregator output. Out of scope for Phase 1-3; track in `.issues/` if the G6 PoC (riir-ai Plan 313) shows CUCG-style rubrics would benefit from Δ unification.

## Phase 5 — Promotion Gate (BLOCKED on riir-ai Plan 313 G6)

### Tasks

- [x] **T5.1** ~~BLOCKED~~ UNBLOCKED — riir-ai Plan 313 Phase 5 G6 quality-parity PoC PASSED (commit riir-ai 0c099b41 + 9e581caf, 2026-07-06): 81.6% drift reduction vs no-attribution baseline (DirectionDelta mutations), 63.1% vs TrajectoryDoctor-only, 895/1000 rollbacks.
- [x] **T5.2** PROMOTED — katgpt-rs commit 55a5184f (2026-07-06): `step_attribution_qualifier` added to root `default` features list. GOAT gate G2 now PASS.
- [-] **T5.3** N/A — PoC PASSED, so the REFUTE path is not taken. The initial REFUTE (Failure mutations, Δ=0 degenerate) and the redesigned PASS (DirectionDelta mutations) are both preserved in riir-ai/.benchmarks/313_step_attribution_goat.md per §3.6.

## GOAT Gate Summary

| Gate | Evidence | Status |
|------|----------|--------|
| **G1** Correctness (Δ≥0 logic; localize_and_link returns correct fault) | Phase 2 unit tests | ✅ PASS — 14/14 green (2026-07-06) |
| **G2** Quality-parity (reproduces SkillAdaptor ±8.1→±5.2 variance reduction) | riir-ai Plan 313 Phase 5 G6 PoC | ✅ PASS — 81.6% drift reduction vs baseline (DirectionDelta mutations, riir-ai commit 0c099b41) |
| **G3** No-regression (feature off = byte-identical to develop) | Phase 2 + 4 isolation tests | ✅ PASS — module is cfg-gated; default-features check clean |
| **G4** Perf (gate overhead < 1µs at W=64, excluding executor) | Phase 3 bench | ✅ PASS — 13 ns aggregate-only @ W=64 (76.9× margin) |
| **G5** Modelless (no riir-train/riir_gpu/backprop dep) | Cargo.toml dep audit | ✅ by construction (no new deps) |
| **G6** Feature-isolation (single-feature + all-features check clean) | `cargo check --features step_attribution_qualifier` + `--all-features` | ✅ PASS — single-feature + root passthrough + `--all-features` (37.25s) all clean |

## Failure Mode

If the trait shape (especially `CandidateMutation::apply_to` returning `K` by value) proves too rigid for the riir-ai consumer (which needs `&mut BranchBank` in-place mutation for hot-path compatibility):
1. Add a parallel `InPlaceCandidateMutation<K>` trait with `fn apply_in_place(&self, baseline: &mut K)`.
2. Add `qualify_in_place()` method mirroring `qualify()` but using the in-place trait.
3. Do NOT remove the by-value trait — it's the cleaner API for generic consumers.

Track in `.issues/` if this surfaces during riir-ai Plan 313 Phase 1 wiring.
