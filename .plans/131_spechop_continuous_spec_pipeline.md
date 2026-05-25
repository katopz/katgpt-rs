# Plan 131: SpecHop — Continuous Multi-Hop Speculation Pipeline

> **Research:** [091 — SpecHop Continuous Multi-Hop Speculation](../.research/091_SpecHop_Continuous_Multi_Hop_Speculation.md)
> **Paper:** [arXiv:2605.21965](https://arxiv.org/pdf/2605.21965) — Continuous speculation for multi-hop retrieval agents
> **Feature Gate:** `spechop` (**Opt-in**, requires GOAT proof before default-on promotion)
> **Depends on:** Plan 030 (Bandit), speculative module (DDTree + verifier), Plan 112 (SR²AM configurator)
> **Status:** ✅ Phase 1–7 + T40-T42 Complete (T1–T32, T40-T42) · Phase 8-9 remaining

## Summary

Implement continuous speculation at **hop/trajectory level** (not token level). SpecHop maintains k speculative threads that predict tool-call observations ahead of actual tool responses. When the target tool returns, a verifier checks equivalence → commit correct branch, rollback incorrect ones. Theoretical framework (α, β, p) gives principled thread-count sizing via k* = ⌈(1+β)/(α+β)⌉.

Our existing DDTree operates at **token granularity**. This plan extends speculation to **tool-call (hop) granularity** — predicting entire observations while the LLM continues reasoning. The commit/rollback pattern maps to our DDTree branch management. The cost model integrates with SR²AM configurator (Plan 112).

**Target: 25–40% wall-clock latency reduction on multi-hop tool-use trajectories, lossless under verifier.**

---

## Tasks

### Phase 1: Core Types & Cost Model
- [x] **T1**: Create `src/spechop/mod.rs` — module index, re-exports, `#[cfg(feature = "spechop")]` gate
- [x] **T2**: Create `src/spechop/types.rs` — `SpecHopConfig`, `HopObservation`, `SpecOutcome` enum
- [x] **T3**: Implement `SpecHopConfig` with α (relative speculator latency), β (decode-to-tool ratio), p (speculator accuracy), k (thread count), auto-compute k* from α and β
- [x] **T4**: Create `src/spechop/cost_model.rs` — `compute_optimal_k(α, β) → usize`, `oracle_rel_lat(α, β, p) → f64`, `bounded_rel_lat(α, β, p, k) → f64`, `starvation_prob(k, α, β, ν) → f64` (Theorem 4)
- [x] **T5**: Unit tests: k*=⌈(1+β)/(α+β)⌉ matches paper examples (α=0.2,β=0.15→k≈4; α=0.3,β=0.75→k≈2), RelLat formula matches paper Table 1
- [x] **T6**: Add `spechop = ["bandit"]` feature gate to `Cargo.toml`, add `pub mod spechop` to `lib.rs`

### Phase 2: Observation Verifier
- [x] **T7**: Create `src/spechop/verifier.rs` — `ObservationVerifier` trait with `verify(o_target: &str, o_spec: &str) -> bool`
- [x] **T8**: Implement `RuleBasedVerifier` — normalize, check refusal patterns, numeric consistency, Jaccard ≥ 0.55, substring match, short-answer exact match (paper Appendix D.4)
- [x] **T9**: Implement `TokenSetJaccard` helper — remove stopwords, compute token-set Jaccard similarity
- [x] **T10**: Unit tests: identical observations → true, different numbers → false, paraphrased → true (Jaccard ≥ 0.55), short answer mismatch → false, refusal pattern → false

### Phase 3: Hop Speculator Trait & Implementations
- [x] **T11**: Create `src/spechop/speculator.rs` — `HopSpeculator` trait with `speculate(action: &str) -> Result<String, SpecError>`
- [x] **T12**: Implement `CacheSpeculator` — HashMap-based cache lookup, returns cached observation or error (modelless path)
- [x] **T13**: Implement `BanditSpeculator<P: ScreeningPruner>` — uses bandit Q-values to predict high-relevance observations (modelless-model-based bridge)
- [x] **T14**: Unit tests: CacheSpeculator hit/miss, BanditSpeculator delegates to ScreeningPruner relevance

### Phase 4: Spec Window Manager
- [x] **T15**: Create `src/spechop/window.rs` — `SpecWindow` struct managing up to k speculative threads
- [x] **T16**: Implement `SpecWindow::push_thread()` — add new speculative thread, panic if > k
- [x] **T17**: Implement `SpecWindow::verify_earliest()` — verify oldest pending thread, return `SpecOutcome::Commit` or `SpecOutcome::Rollback`
- [x] **T18**: Implement `SpecWindow::rollback_all()` — discard all speculative work, reset to last verified state
- [x] **T19**: Unit tests: window capacity enforcement, commit shifts window, rollback clears downstream, sequential commits advance state

### Phase 5: Continuous Pipeline Loop
- [x] **T20**: Create `src/spechop/pipeline.rs` — `SpecHopPipeline` struct with config, speculator, verifier, window
- [x] **T21**: Implement `SpecHopPipeline::execute()` — main loop: extend window → verify earliest → repeat (Algorithm 1 from paper)
- [x] **T22**: Implement hop-level state machine: `HopState::AwaitingTarget` | `Speculating` | `Committed` | `RolledBack`
- [x] **T23**: Implement early termination: if verified thread reaches final answer, return immediately
- [x] **T24**: Integration test: synthetic 4-hop trajectory with cache speculator, verify final answer matches sequential execution

### Phase 6: DDTree Integration
- [x] **T25**: Add `SpecHopMode` variant to DDTree builder — when enabled, DDTree branches represent speculative hops (not tokens)
- [x] **T26**: Implement `build_dd_tree_spechop()` — hop-level DDTree where each node is a (action, observation) pair, branch score = speculator confidence
- [x] **T27**: Wire `ObservationVerifier` into DDTree verification path — accept/reject branches at hop granularity
- [x] **T28**: Integration test: DDTree with spechop produces same best-path as sequential DDTree when speculator is perfect (p=1.0)

### Phase 7: SR²AM Configurator Integration
- [x] **T29**: Add `PlanningDecision::SpecHop { k: usize }` arm to SR²AM configurator (Plan 112)
- [x] **T30**: Implement auto-k computation: measure α and β from recent inference stats, compute k* via cost model
- [x] **T31**: Implement configurator reward: `reward = latency_reduction / compute_overhead` — only activate spechop when ratio > 1.0
- [x] **T32**: Unit test: configurator selects SpecHop when α < 0.3 and β < 0.5 (tool-bound scenarios), skips when β > 0.8 (decode-bound)

### Phase 8: GOAT Proof (6/6 Required for Default-On Consideration)
- [ ] **T33**: Proof 1 — Losslessness: run Bomber arena 1000 rounds with and without spechop, identical win rates (±2%), identical game traces (verified via EventLog)
- [ ] **T34**: Proof 2 — Latency reduction: measure wall-clock time on 4-hop synthetic trajectory, spechop achieves RelLat within 15% of theoretical RelLat*
- [ ] **T35**: Proof 3 — Thread starvation: P_starve < 5% with measured α, β, k (Theorem 4 bound)
- [ ] **T36**: Proof 4 — Cache-as-speculator: CacheSpeculator with 25% cache achieves p̂ ≥ 0.3 on synthetic retrieval task
- [ ] **T37**: Proof 5 — Compute overhead: total tool+speculator calls ≤ 2× sequential calls (bounded compute cost)
- [ ] **T38**: Proof 6 — Compatibility: test with `bandit`, `bt_rank`, `spectral_quant`, `dash_attn` feature combinations. No panics, no NaN

### Phase 9: Benchmarks & Documentation
- [ ] **T39**: Create `.benchmarks/038_spechop_goat.md` — all 6 GOAT proof results with commands to reproduce
- [x] **T40**: Add `spechop_01_pipeline` example — demonstrate 4-hop continuous speculation with cache speculator
- [x] **T41**: Add `spechop_02_cost_model` example — show α/β/p → k* computation and RelLat prediction
- [x] **T42**: Update `README.md` — add SpecHop section under speculative pipeline, document feature gate, link to benchmark results
- [x] **T43**: Update `.docs/` — add architecture diagram showing hop-level speculation flow
- [ ] **T44**: Commit with message: `feat(spechop): continuous multi-hop speculation pipeline (Plan 131)`

---

## Architecture

```
src/spechop/
├── mod.rs              # Module index, re-exports, feature gate
├── types.rs            # SpecHopConfig, HopObservation, SpecOutcome, HopState
├── cost_model.rs       # α/β/p → k* computation, RelLat formulas (Theorems 2, 4)
├── verifier.rs         # ObservationVerifier trait + RuleBasedVerifier
├── speculator.rs       # HopSpeculator trait + CacheSpeculator + BanditSpeculator
├── window.rs           # SpecWindow thread pool manager (commit/rollback)
├── pipeline.rs         # SpecHopPipeline continuous loop (Algorithm 1)
├── hop_tree.rs         # Hop-level DDTree integration (T25–T27)
└── tests.rs            # Integration tests
```

### Key Enums

```rust
/// Outcome of verifying a speculative thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecOutcome {
    /// Speculative observation matches target → commit branch.
    Commit,
    /// Speculative observation differs → rollback to verified state.
    Rollback,
}

/// State of a single hop in the speculative pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HopState {
    /// Waiting for target tool to return.
    AwaitingTarget,
    /// Speculator has predicted observation, LLM continuing.
    Speculating,
    /// Verification passed, observation committed.
    Committed,
    /// Verification failed, rolled back.
    RolledBack,
}
```

### Key Structs

```rust
/// Configuration for SpecHop pipeline.
/// Parameters from paper Section 3.
#[derive(Clone, Debug)]
pub struct SpecHopConfig {
    /// Relative speculator latency: E[T_spec] / E[T_target]. Must be < 1.0.
    pub alpha: f64,
    /// Decode-to-tool ratio: E[T_seg] / E[T_target].
    pub beta: f64,
    /// Speculator success probability per hop.
    pub p: f64,
    /// Maximum active speculative threads. None = auto-compute from α, β.
    pub k: Option<usize>,
    /// Volatility bound for starvation probability. Default: 0.4.
    pub volatility: f64,
}

/// A single hop observation pair (target vs speculative).
#[derive(Clone, Debug)]
pub struct HopObservation {
    /// The action that triggered this hop (e.g., a search query).
    pub action: String,
    /// Target tool observation (may be pending).
    pub o_target: Option<String>,
    /// Speculative observation (from speculator S).
    pub o_spec: Option<String>,
    /// Current state of this hop.
    pub state: HopState,
}
```

### Cost Model (Theorem 4)

```rust
impl SpecHopConfig {
    /// Compute optimal thread count: k* = ⌈(1 + β) / (α + β)⌉
    pub fn optimal_k(&self) -> usize {
        let k_det = (1.0 + self.beta) / (self.alpha + self.beta);
        k_det.ceil() as usize
    }

    /// Oracle relative latency upper bound: RelLat* = 1 - p(1-α)/(1+β)
    pub fn oracle_rel_lat(&self) -> f64 {
        1.0 - self.p * (1.0 - self.alpha) / (1.0 + self.beta)
    }

    /// Bounded-window relative latency: RelLat_k
    pub fn bounded_rel_lat(&self, k: usize) -> f64 {
        let mu_k = (1.0 - self.p.powi(k as i32)) / (1.0 - self.p);
        1.0 - (1.0 - self.alpha) * (1.0 - (1.0 - self.p) / mu_k) / (1.0 + self.beta)
    }

    /// Pipeline starvation probability bound (Theorem 4, CLT approximation).
    pub fn starvation_prob(&self, k: usize) -> f64 {
        // Φ((1+β - k(α+β)) / (ν * sqrt(k*α² + (k-1)*β² + 1)))
        let numerator = (1.0 + self.beta) - k as f64 * (self.alpha + self.beta);
        let variance = k as f64 * self.alpha.powi(2)
            + (k as f64 - 1.0) * self.beta.powi(2)
            + 1.0;
        let z = numerator / (self.volatility * variance.sqrt());
        normal_cdf(z)
    }
}
```

---

## Feature Gate

```toml
# Cargo.toml
spechop = ["bandit"]  # Continuous multi-hop speculation pipeline (Plan 131)
```

```rust
// lib.rs
#[cfg(feature = "spechop")]
pub mod spechop;
```

**Not in default features** until GOAT 6/6 proved.

---

## Compatibility Matrix

| Feature | Compatible | Notes |
|---------|-----------|-------|
| `bandit` | ✅ Required | BanditPruner feeds into speculator decisions |
| `bt_rank` | ✅ | Bradley-Terry ranking for branch selection |
| `spectral_quant` | ✅ | KV cache compression orthogonal |
| `dash_attn` | ✅ | Sparse attention + hop speculation complementary |
| `rt_turbo` | ✅ | Retrieval heads can serve as hop speculators |
| `sr2am_configurator` | ✅ | Configurator decides k (thread count) |
| `data_gate` | ✅ | Data gating for training, spechop for inference |
| `lt2_looped` | ⚠️ Test | Looped inference may interact with hop-level speculation |
| `dllm` / `dmax_spd` | ⚠️ Test | Diffusion speculation + hop speculation may conflict |
| `game_state` | ✅ | Game forward model as "target tool" for hop speculation |

---

## Expected Outcome

If GOAT proofs pass (6/6):
- Promote `spechop` to default-on candidate (requires separate audit)
- Expected latency reduction: 25–40% on multi-hop tool-use trajectories
- Add to production stack in README tech table
- Integrate k* auto-computation into SR²AM configurator

If GOAT proofs fail:
- Keep as opt-in `spechop` feature gate
- Document negative result in `.benchmarks/`
- Record which proofs failed and why
- Still valuable: cost model (α/β/p) is useful for theoretical analysis regardless

---

## GOAT Proof Criteria

1. **Losslessness**: Identical game traces with and without spechop ✓
2. **Latency reduction**: RelLat within 15% of theoretical prediction ✓
3. **Starvation bound**: P_starve < 5% with measured parameters ✓
4. **Cache speculator**: p̂ ≥ 0.3 with 25% cache ✓
5. **Compute overhead**: ≤ 2× total tool+speculator calls vs sequential ✓
6. **Compatibility**: No panics/NaN with default feature combinations ✓

---

## Implementation Order

```
Phase 1: Core types & cost model       [~2h]
Phase 2: Observation verifier           [~2h]
Phase 3: Hop speculator trait           [~2h]
Phase 4: Spec window manager            [~3h]
Phase 5: Continuous pipeline loop       [~3h]  ← main work
Phase 6: DDTree integration             [~2h]
Phase 7: SR²AM configurator             [~2h]
Phase 8: GOAT proofs                    [~3h]
Phase 9: Benchmarks & docs              [~2h]
───────────────────────────────────────────────
Total estimate:                         ~21h
```

---

## Dependencies

- `bandit` feature — BanditPruner for speculator decisions
- `speculative` module — DDTree branch management, verifier pattern
- `katgpt-core` traits — ConstraintPruner, ScreeningPruner
- No new external crates required

---

## References

- SpecHop paper: https://arxiv.org/pdf/2605.21965
- Speculative Actions (predecessor): https://arxiv.org/abs/2510.04371
- Speculative Decoding (Leviathan et al.): https://arxiv.org/abs/2302.01318
- Our speculative infrastructure: `src/speculative/`
- Our REAP duality mapping: Research 037
- Our SR²AM configurator: Research 076, Plan 112
- Our RTPurbo retrieval heads: Research 086, Plan 126
- Our cost model (MoE+SD): Research 059, Plan 096