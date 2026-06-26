# Plan 332: Thinking-to-Recall — Intermediate-Fact-Verified Trajectory Gate

**Date:** 2026-06-26
**Research:** [katgpt-rs/.research/313_Thinking_To_Recall_Intermediate_Fact_Verified_Trajectories.md](../.research/313_Thinking_To_Recall_Intermediate_Fact_Verified_Trajectories.md)
**Source paper:** [arXiv:2603.09906](https://arxiv.org/abs/2603.09906) — Gekhman et al. 2026, "Thinking to Recall"
**Target:** `crates/katgpt-core/src/` (new module `intermediate_fact_gate/`) + Cargo feature `intermediate_fact_gate`
**Status:** Active — Phase 0 (skeleton)

---

## Goal

Ship a modelless **intermediate-fact-verified trajectory gate** that composes four already-shipped primitives — BoMSampler (k-trajectory sampling), Engram/anchor extraction (intermediate facts), FaithfulnessProbe (per-fact causal verification against committed memory), and CLR (final-claim reliability vote) — into a two-stage test-time trajectory selector.

The gate distills paper §"Building more reliable models": generate multiple trajectories, retain only those whose intermediate facts are verifiably hallucination-free, then vote the survivors. The strict-AND filter (one bad intermediate fact kills the trajectory) is the paper's headline empirical protocol.

**GOAT gate (must pass before any default promotion):**
- **G1** — correctness: gate retains all ground-truth-verified trajectories and discards all trajectories with at least one synthetically-corrupted intermediate fact (deterministic unit tests).
- **G2** — ablation: gate's accuracy ≥ CLR-only baseline on a synthetic trajectory set with planted hallucinations.
- **G3** — no regression: `cargo test -p katgpt-core --features intermediate_fact_gate --lib` clean; no other feature regresses.
- **G4** — latency: gate adds ≤ K × N × t_probe overhead per CLR vote, where K=trajectories, N=intermediate facts per trajectory, t_probe=FaithfulnessProbe latency (~100µs/segment at audit cadence). Target: ≤ 1 ms total per entity per decision when K=8, N=4.

**Promotion rule (per AGENTS.md GOAT gate discipline):** stays opt-in (`intermediate_fact_gate = ["faithfulness_probe", "clr", "engram", "bom_sampler"]`) until G1–G4 pass on a synthetic planted-hallucination benchmark. Demote the loser (CLR-only baseline) if the gated version wins on accuracy at parity compute.

---

## Scope

**In scope (this plan):**
- New `intermediate_fact_gate` module under `crates/katgpt-core/src/`.
- New trait `IntermediateFactExtractor` (extract intermediate facts from a trajectory).
- New struct `IntermediateFactGate` composing `FaithfulnessProbe` + CLR vote.
- Synthetic benchmark with planted hallucinations (G2).
- Latency benchmark (G4).

**Out of scope (deferred to riir-ai):**
- Per-NPC runtime wiring at 20Hz tick (slots into Cognitive Integrity Layer loop step 6, see `riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md`).
- Game-specific fact extractors (combat, dialog, faction).
- Integration with live inference pipeline (Bomber/Go) — requires the G6 deferred Engram integration too.

**Not in scope (→ riir-train):**
- Process-reward training (paper §"Building more reliable models" training recipe).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/intermediate_fact_gate/mod.rs` with module doc citing paper §"Building more reliable models" and Research 313.
- [ ] **T1.2** Define the `IntermediateFactExtractor` trait:

```rust
/// Extract intermediate "facts" from a sampled trajectory.
///
/// A "fact" is a modelless anchor: a direction vector, an Engram hash address,
/// or a KG triple emitted along the trajectory. NOT a decoded token string.
/// Distillation of paper §"Mechanism 2: Factual priming" + Research 313 §2.3.
pub trait IntermediateFactExtractor<Trajectory> {
    type Fact;
    type FactsIter<'a>: Iterator<Item = &'a Self::Fact> where Self::Fact: 'a;

    /// Extract the ordered list of intermediate facts along the trajectory.
    /// Order matters: the AND-gate reports *which* fact failed.
    fn extract_facts<'a>(&'a self, trajectory: &'a Trajectory) -> Self::FactsIter<'a>;
}
```

- [ ] **T1.3** Define the gate struct and the AND-gate verdict:

```rust
/// Per-trajectory verdict from the intermediate-fact gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactGateVerdict {
    /// All intermediate facts passed FaithfulnessProbe::is_faithfully_used.
    Retained,
    /// At least one intermediate fact failed verification.
    /// `failed_index` is the position of the first failure in extraction order.
    Discarded { failed_index: usize },
}

/// The gate. Composes an extractor + a faithfulness probe + a retain threshold.
///
/// Modelless: zero training, zero backprop. Verification reads frozen committed
/// memory only.
pub struct IntermediateFactGate<E, P> {
    extractor: E,
    probe: P,
    /// Threshold passed to `FaithfulnessProfile::is_faithfully_used`.
    /// Per Research 313 §2.3, this is a scalar sigmoid-gate threshold on the
    /// behavioral delta. Lower = stricter (more trajectories discarded).
    retain_threshold: f32,
}

impl<E, P> IntermediateFactGate<E, P> {
    pub fn new(extractor: E, probe: P, retain_threshold: f32) -> Self { /* ... */ }

    /// Run the AND-gate over all intermediate facts in `trajectory`.
    /// Strict-AND: returns Discarded on the FIRST failed fact (paper §1.4).
    pub fn verify_trajectory<T>(&self, trajectory: &T) -> FactGateVerdict
    where
        E: IntermediateFactExtractor<T>,
        P: FaithfulnessProbe<Memory = E::Fact>,
        // ... + ConsumerContext bound for the probe call
    { /* ... */ }
}
```

- [ ] **T1.4** Define the batch-level filter that runs the gate over k trajectories and partitions into `(retained, discarded)`:

```rust
/// Partition k sampled trajectories into (retained, discarded) by the AND-gate.
///
/// Distillation of paper §"Building more reliable models": retain only
/// trajectories whose intermediate facts are verifiably hallucination-free.
/// Caller (typically CLR runtime) then votes over `retained`.
pub fn filter_trajectories<E, P, T>(
    gate: &IntermediateFactGate<E, P>,
    trajectories: &[T],
) -> (Vec<usize>, Vec<(usize, FactGateVerdict)>)
/* returns (retained_indices, discarded_with_verdicts) */
```

- [ ] **T1.5** Add the Cargo feature `intermediate_fact_gate` to `crates/katgpt-core/Cargo.toml`:
  ```toml
  intermediate_fact_gate = ["faithfulness_probe", "engram"]
  ```
  (CLR and BoMSampler compose via the *caller*, not via feature dependency — the gate itself only needs FaithfulnessProbe + Engram.)

- [ ] **T1.6** Add unit tests for the gate mechanics: `verify_trajectory` returns `Retained` when all facts pass; returns `Discarded { failed_index: i }` on the first failure.

**Exit:** `cargo test -p katgpt-core --features intermediate_fact_gate --lib intermediate_fact_gate::` passes (5+ tests).

---

## Phase 2 — FactAnchor vs FillerAnchor (M2 quality gate)

Distillation of paper §"Mechanism 2: Factual priming" — facts alone (strict filtering of filler) recover most of CoT's gain. This phase adds the anchor-selection quality gate.

### Tasks

- [ ] **T2.1** Define `AnchorKind` enum distinguishing hard-fact anchors from filler:

```rust
/// Distinguishes hard-fact anchors (paper §1.3 "concrete facts") from
/// filler anchors (paper §1.3 "filler text, search plans").
///
/// Distillation: paper's strict-filter experiment shows fact-only conditioning
/// recovers most of CoT's gain. Anchor selection should prefer FactAnchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorKind {
    /// Topically-grounded hard fact (KG triple, named entity, committed shard hash).
    Fact,
    /// Filler text, search plans, target-answer restatements.
    /// Per paper §1.3, these should be filtered at extraction time, not fusion time.
    Filler,
}
```

- [ ] **T2.2** Extend `IntermediateFactExtractor` to optionally tag each fact with `AnchorKind`. Provide a default impl that returns `Fact` for all (backward-compatible).
- [ ] **T2.3** Add a `filler_filter` helper: `filter_filler(facts) -> impl Iterator<Item=Fact>` that drops `AnchorKind::Filler` entries. Caller can wrap the extractor with this to enforce fact-only priming (paper §1.3 protocol).
- [ ] **T2.4** Unit tests: `filler_filter` drops only `Filler`-tagged entries; ordering preserved.

**Exit:** M2 distillation shipped as a composable anchor-quality layer; Engram/OctreeCTC consumers can opt in.

---

## Phase 3 — Synthetic Planted-Hallucination Benchmark (G1 + G2)

### Tasks

- [ ] **T3.1** Build a synthetic trajectory fixture: K=8 trajectories, each with N=4 intermediate facts. Plant one hallucinated fact at a known position in 4 of the 8 trajectories. The other 4 are clean.
- [ ] **T3.2** G1 correctness test: gate retains exactly the 4 clean trajectories, discards exactly the 4 planted ones, with correct `failed_index` for each discarded trajectory.
- [ ] **T3.3** G2 ablation test: compare (a) CLR-only vote over all 8 trajectories vs (b) gate-then-CLR vote. The gated version must score higher accuracy on a final-claim reliability metric (because the planted-hallucination trajectories drag down CLR-only mean reliability). Document the margin.
- [ ] **T3.4** Negative control: when all trajectories are clean (no planted hallucinations), gate's accuracy equals CLR-only (no false discards). Verifies the gate doesn't over-filter.

**Exit:** G1 + G2 pass on synthetic data. Record results in `.benchmarks/332_intermediate_fact_gate_goat.md`.

---

## Phase 4 — Latency Benchmark (G4)

### Tasks

- [ ] **T4.1** Add a latency bench: K=8 trajectories × N=4 facts × FaithfulnessProbe cost. Measure total gate overhead per entity per decision. Target: ≤ 1 ms.
- [ ] **T4.2** If G4 fails (>1 ms), document the root cause (likely FaithfulnessProbe audit-cadence cost). Mitigation: the probe already ships an audit-cadence mode (`DefaultFaithfulnessProbe` runs probes periodically, not every tick, per Plan 278 T2.9). Document that the gate inherits the audit cadence — at steady state, only K × (audit_fraction) trajectories get probed per tick.
- [ ] **T4.3** Document the steady-state cost model: per-tick gate cost = `K × audit_fraction × N × t_probe`. With `audit_fraction = 0.1`, `K=8`, `N=4`, `t_probe = 100µs`: steady-state ≈ 320 µs per entity per tick. Within 20Hz tick budget (50 ms) with ~150× headroom for crowd-scale.

**Exit:** G4 passes (or fails with documented mitigation via audit cadence).

---

## Phase 5 — GOAT Gate Decision

### Tasks

- [ ] **T5.1** Run all four gates (G1 correctness, G2 ablation, G3 no-regression, G4 latency).
- [ ] **T5.2** Record verdict in `.benchmarks/332_intermediate_fact_gate_goat.md`:
  - If all gates pass AND G2 margin > 0 → **promote to default-on** in a follow-up PR (add `intermediate_fact_gate` to the `default` feature list). Demote the CLR-only baseline as the loser.
  - If G2 margin ≤ 0 (gate doesn't beat CLR-only on accuracy) → keep opt-in, document that the gate is correct but provides no quality gain on synthetic data; defer to riir-ai runtime integration for real-workload evidence.
  - If G4 fails AND audit-cadence mitigation is insufficient → keep opt-in, file an issue for FaithfulnessProbe hot-path optimization.
- [ ] **T5.3** Update `katgpt-rs/.docs/01_overview.md` Feature Flags table with the new feature + GOAT verdict.
- [ ] **T5.4** Update `katgpt-rs/README.md` Feature Showcase with a one-paragraph entry (only if promoted to default).

**Exit:** GOAT gate decision recorded. Either promoted to default, or opt-in with documented evidence.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| FaithfulnessProbe latency blows the 1ms G4 budget | Probe already ships audit-cadence mode (Plan 278 T2.9); gate inherits it. Steady-state cost is `K × audit_fraction × N × t_probe`, not `K × N × t_probe` per tick. |
| Synthetic benchmark doesn't reflect real NPC trajectories | G2 is a *lower bound* on accuracy gain. Real-workload evidence is a riir-ai responsibility (Cognitive Integrity Layer runtime integration). The open primitive ships correct + fast; the private runtime proves the gain. |
| Strict-AND over-filters (discards trajectories that have one bad fact but a correct final answer) | This matches the paper's headline finding ("a single hallucinated intermediate fact degrades the final answer"). If real-workload evidence contradicts, add a `max_failed_facts` parameter (default 0 = strict-AND). |
| Anchor extraction is game-specific | The trait is generic; concrete extractors are caller concerns. Ship one default extractor that wraps Engram hash addresses (modelless, no game semantics). |
| The gate might compose poorly with CLR's `(mean)^M` (CLR assumes all K trajectories vote; gate removes some) | CLR's reliability vote is normalized over the *retained* set, not the original K. Document in the integration doc: "CLR operates on `retained.len()` trajectories, not `K`." If `retained.is_empty()`, fall back to CLR-over-all (degrade to baseline rather than emit no action). |

---

## Cross-references

- **Research note:** [katgpt-rs/.research/313_Thinking_To_Recall_Intermediate_Fact_Verified_Trajectories.md](../.research/313_Thinking_To_Recall_Intermediate_Fact_Verified_Trajectories.md)
- **FaithfulnessProbe (dependency):** [Plan 278](278_faithfulness_probe_modelless.md)
- **CLR (composer):** [Plan 284](284_runtime_clr_self_adaptive_loop.md)
- **BoMSampler (composer):** [Plan 281](281_bom_single_pass_diverse_sampling.md)
- **Engram (anchor source):** [Plan 299](299_Engram_Hash_Addressed_Pattern_Memory.md)
- **PathConsistency (training-time analog):** [Plan 054](054_stepcode_reasoner_modelless.md)
- **Cognitive Integrity Layer (runtime home):** `riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md` + `riir-ai/.plans/308_cognitive_integrity_layer.md`
- **Source paper:** [arXiv:2603.09906](https://arxiv.org/abs/2603.09906)

---

## TL;DR

Ship a modelless **intermediate-fact-verified trajectory gate** behind `intermediate_fact_gate = ["faithfulness_probe", "engram"]`. The gate composes BoMSampler (k trajectories) → IntermediateFactExtractor (Engram anchors along each trajectory) → FaithfulnessProbe (per-fact AND-gate) → CLR vote (over retained trajectories). Strict-AND matches the paper's headline finding. GOAT gate: G1 correctness (retain clean / discard planted), G2 ablation (gated accuracy > CLR-only), G3 no-regression, G4 latency (≤1ms per entity). Promote to default if all gates pass and G2 margin > 0. M2 distillation (`FactAnchor` vs `FillerAnchor`) ships as Phase 2 — unblocks Engram G6 ("do fact-only anchors bind better than mixed?"). M1 (compute buffer) is validation only, no code. Runtime wiring deferred to riir-ai Cognitive Integrity Layer (the gate slots into the existing integrity loop at step 6).
