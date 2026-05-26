# Plan 142: State-Source Modelless Distillation

**Research:** 103 (State Distribution View of Post-Training)
**Status:** ✅ COMPLETE
**Feature Gate:** `state_source` (off by default, opt-in GOAT proof)
**Depends On:** `bandit`
**Domain:** katgpt-rs (modelless core)

---

## Motivation

Paper (arXiv:2605.22731) proves that **where updates are applied** (state distribution) matters as much as **what signal is provided** (objective). Key result: OPD from a degraded teacher surpasses that teacher on all metrics — because the student controls state visitation while receiving local guidance.

Our modelless stack already has on-policy components (`BanditPruner`, `DeltaBanditPruner`) but lacks:
1. **State-visitation tracking** — we don't measure coverage entropy
2. **OPD-style state-source separation** — no component where learner controls states, validator provides local guidance
3. **Retention-aware GOAT proofs** — we measure throughput/accuracy but not action-diversity preservation

This plan adds these three capabilities as a single feature-gated module.

---

## Distillation Targets

### D1: State-Source Taxonomy (Documentation)

Map our existing components to the paper's two-axis framework. This is research documentation, no code change.

| Component | State Source | Signal Source | Paper Analogy |
|-----------|-------------|---------------|---------------|
| `AbsorbCompress` | Dataset (off-policy) | Binary outcome | SFT-like |
| `BanditPruner` | Learner rollouts (on-policy) | δ-reward / win-loss | RL-like |
| `DeltaBanditPruner` | Learner rollouts (on-policy) | Hint-δ | RL (dense reward) |
| `FlowPruner` | Dataset (off-policy) | Flow bonus from marginal | Offline KD-like |
| VPD M-step | Learner Q-values (on-policy) | Teacher KL | OPD-like |
| SDAR gate | Same as wrapped pruner | Sigmoid-modulated | Signal-only modification |

### D2: State-Visitation Entropy Tracker

Add `StateVisitationTracker` — tracks prefix-state coverage and reports entropy.

```rust
/// Tracks state visitation distribution during bandit rollouts.
/// Reports entropy-based coverage metrics. Zero-cost when disabled.
pub struct StateVisitationTracker {
    /// Hash-based state counter (blake3 prefix hash → visit count)
    visits: HashMap<u64, u32>,
    /// Total visits for entropy computation
    total: u64,
    /// Coverage threshold for exploration boost signal
    coverage_threshold: f32,
}

impl StateVisitationTracker {
    /// Record a visited state (prefix hash).
    /// O(1) amortized via HashMap.
    pub fn observe(&mut self, prefix_hash: u64);

    /// Compute visitation entropy H = -Σ p(s) log p(s).
    /// Higher = more diverse exploration.
    pub fn entropy(&self) -> f32;

    /// Is coverage above threshold?
    /// When false → suggest exploration boost to BanditPruner.
    pub fn coverage_ok(&self) -> bool;

    /// Number of unique states visited.
    pub fn unique_states(&self) -> usize;
}
```

**Integration point:** `BanditPruner::select_action()` calls `tracker.observe(prefix_hash)`. When `!tracker.coverage_ok()`, boost ε in ε-greedy or increase UCB1 exploration constant.

**GOAT proof targets:**
- `entropy()` computation: ≤1µs for 10K unique states
- `observe()`: ≤50ns (hashmap insert)
- Zero overhead when feature disabled (compile-time gate)

### D3: Validator-Continuation Scoring (Modelless OPD Analogue)

The paper's breakthrough: continuation-based OPD where teacher generates short rollouts from student states. For modelless: **WASM validator provides valid continuation sequences from student-visited states**.

```rust
/// Modelless OPD analogue: validator provides short validated continuations
/// from student-sampled states. Student controls state visitation (on-policy),
/// validator provides local trajectory guidance (dense supervision).
pub struct ValidatorContinuationScorer<V: ConstraintPruner> {
    validator: V,
    /// Maximum continuation depth
    max_depth: usize,
    /// Number of candidate continuations to sample
    n_candidates: usize,
}

impl<V: ConstraintPruner> ValidatorContinuationScorer<V> {
    /// From a student state (prefix), generate short valid continuations
    /// using the constraint pruner as guide. Returns scored continuations.
    ///
    /// This is the modelless analogue of OPD: student controls the state
    /// (it picked the prefix), validator provides local guidance (which
    /// continuations are valid).
    pub fn score_continuations(
        &self,
        prefix: &[usize],
        n_actions: usize,
    ) -> Vec<ContinuationScore>;

    /// Score a DDTree beam by its valid-continuation density.
    /// Beams with more valid continuations get a bonus.
    pub fn beam_score(&self, beam: &[usize], base_score: f32) -> f32;
}

pub struct ContinuationScore {
    /// The continuation tokens
    pub tokens: Vec<usize>,
    /// Fraction of valid continuations found (0..1]
    pub valid_density: f32,
    /// Depth reached before dead-end
    pub reachable_depth: usize,
}
```

**Note:** For game domains, this is super-GOAT — the WASM validator encodes game-specific knowledge that nobody else has. This is Pillar 2 (WASM Validators) from the Decision Matrix being used as an OPD-style local guide. The strategic value (student surpasses teacher even with degraded LoRA) makes this a competitive advantage.

**For katgpt-rs:** Generic `ConstraintPruner`-based implementation. Game-specific WASM validators live in riir-ai.

### D4: GOAT Retention Dimension

Add action-diversity preservation to GOAT proof checklist. New metric:

```rust
/// Retention metric for GOAT proofs: does method X preserve Y% of
/// baseline action diversity?
pub struct RetentionMetric {
    /// Baseline action distribution entropy
    baseline_entropy: f32,
    /// Post-training action distribution entropy
    post_entropy: f32,
    /// Retention ratio (post/baseline, 1.0 = perfect retention)
    pub retention_ratio: f32,
}

impl RetentionMetric {
    /// Compute from two action distributions.
    /// Uses KL divergence: retention = exp(-KL(post || baseline))
    pub fn compute(baseline: &[f32], post: &[f32]) -> Self;
}
```

**GOAT proof addition:** Every modelless distillation plan should report `retention_ratio ≥ 0.95` (matching paper's OPD result). Methods that degrade action diversity (like stress SFT's 0.83) should be flagged.

---

## Module Structure

```
src/pruners/
  state_source/           (new, behind feature gate)
    mod.rs                — pub exports
    visitation.rs         — StateVisitationTracker (D2)
    continuation.rs       — ValidatorContinuationScorer (D3)
    retention.rs          — RetentionMetric (D4)
```

## Feature Gate

```toml
[features]
state_source = ["bandit"]
```

**Off by default.** Not in `default` or `full`. Opt-in for GOAT proof.

**Rationale:** This is a meta-level improvement to the modelless stack. It doesn't change hot-path behavior when disabled (zero compile-time cost). When enabled, it adds state-visitation tracking to bandit rollouts and provides the OPD-analogue scoring API.

## GOAT Proof Targets

| # | Metric | Target |
|---|--------|--------|
| G1 | `StateVisitationTracker::observe()` | ≤50ns |
| G2 | `StateVisitationTracker::entropy()` (10K states) | ≤1µs |
| G3 | `ValidatorContinuationScorer::score_continuations()` | ≤10µs per call |
| G4 | `RetentionMetric::compute()` | ≤100ns |
| G5 | Retention ratio (bomber 1000 games) | ≥0.95 |
| G6 | State coverage with tracker vs without | Same action quality, higher entropy |
| G7 | Zero overhead when disabled | compile-time verified |

## Relationship to Decision Matrix

| Pillar | Impact |
|--------|--------|
| Pillar 2: WASM Validators | **Strengthened** — validators now serve as OPD-style local guides, not just filters |
| Pillar 4: Frame-Sampling | **Compatible** — state visitation tracking works per-frame |

This plan directly supports the Decision Matrix's "heads you win, tails you don't lose" thesis: even if LoRA (Secret A) is degraded, the WASM validator (Secret A2) provides on-policy local guidance that lets the student surpass the teacher. The paper's core result validates this commercially.

## Relationship to riir-ai

| Aspect | katgpt-rs | riir-ai |
|--------|-----------|---------|
| `ConstraintPruner` trait | ✅ Defines trait | — |
| `ValidatorContinuationScorer<V>` | ✅ Generic implementation | — |
| Game-specific WASM continuations | — | ✅ BomberValidator, GoValidator continuations |
| LoRA-based OPD (teacher forward pass) | — | ✅ riir-gpu model-based path |
| State-visitation for MCTS | — | ✅ riir-games Fourier MCTS |

**The game-specific continuation knowledge (which valid move sequences matter for Bomber/Go/TFT) stays private in riir-ai.** The generic `ValidatorContinuationScorer<V: ConstraintPruner>` ships in katgpt-rs.

## Tasks

| # | Task | Scope | Est. |
|---|------|-------|------|
- [x] **T1:** Create `state_source/` module structure (Code, 0.5d)
- [x] **T2:** Implement `StateVisitationTracker` (Code, 0.5d)
- [x] **T3:** Implement `ValidatorContinuationScorer<V>` (Code, 1d)
- [x] **T4:** Implement `RetentionMetric` (Code, 0.5d)
- [x] **T5:** Integrate tracker into `BanditPruner` (Code, 0.5d)
- [x] **T6:** GOAT proof test (7 targets) (Test, 0.5d)
- [x] **T7:** Benchmark: bomber 1000 games with/without (Bench, 0.5d)
- [x] **T8:** Update README + feature gate docs (Docs, 0.5d)

**Total: ~4 days**

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| State-visitation tracking overhead | Low | Low | blake3 hash is fast, HashMap get_mut is amortized O(1) |
| Continuation scoring too slow | Medium | Medium | Cap max_depth=3, n_candidates=5; fail-fast on first invalid |
| Retention metric not discriminative | Low | Low | Even if ratio is always ≥0.95, having the metric is valuable for regression detection |
| No arena win-rate improvement | Medium | Low | Paper's result is about capability retention, not target improvement. Same pattern as SDAR/RMSD negative results |

**Honest assessment:** This plan is unlikely to produce arena win-rate improvements (consistent with SDAR, RMSD negative results). Its value is:
1. **Defensive:** Catches state-coverage regressions early
2. **Strategic:** Validates the "WASM validator as OPD guide" thesis for commercial moat
3. **Research:** Provides the state-source taxonomy for future distillation work

The paper's own experiments are small-scale (one model, one task). But the *framework* is sound and our architecture already aligns with it.
