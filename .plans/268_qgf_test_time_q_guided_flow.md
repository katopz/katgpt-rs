# Plan 268: QGF Test-Time Q-Guided Flow — Modelless Primitive

**Research:** `.research/268_QGF_Test_Time_Q_Guided_Flow.md`
**Paper:** [arXiv:2606.11087](https://arxiv.org/pdf/2606.11087) — Q-Guided Flow (Zhou et al., 2026)
**Status:** 🚧 In Progress — Phase 1 (T1-T3) + F4 adaptive (T7) implemented, tests green
**Branch:** `develop` (no new feature branch per project rules)
**Feature Gates:** `qgf` (parent, default OFF until GOAT proof)
  - `qgf_projector` (F2 — FirstOrderProjector)
  - `qgf_oracle` (F3 — QGradientOracle trait)
  - `qgf_drafter` (F1 — QGuidedDrafter, depends on `qgf_projector` + `qgf_oracle`)
  - `qgf_adaptive` (F4 — VarianceAdaptiveGuidance, depends on `qgf_drafter`)
**Depends On:** Plan 229 (NFCoT FlowScore — QGF unblocks it from MARGINAL), Plan 217 (NextLat belief drafter), existing `SpeculativeGenerator` + `LeoHead` + `FlowFieldCache` + `ActionBridge`
**Unblocks:** Plan 229 (NFCoT FlowScore → potential GOAT promotion)
**GOAT Criteria:** ≥3% first-attempt accuracy gain on Sudoku 9×9, ≥5% speculative acceptance rate gain, < 2% overhead, < 5% off-manifold false-positive rate

---

## Overview

Implement the QGF (Q-Guided Flow) test-time gradient-guidance primitive in `katgpt-core`. The primitive lets any `SpeculativeGenerator` be steered by a Q-gradient oracle during generation — the modelless analogue of the paper's Algorithm 1.

**Core equation (discrete analogue of QGF Alg 1):**
```
At generation step t with prefix p_t and drafter velocity v_t:
  â_1  = project_one_step(p_t, v_t, remaining)     # FIRST-ORDER projection
  g    = q_oracle.gradient_at(state, â_1)           # ∇_{â_1} Q (DROP JACOBIAN)
  p'   = p_t + (1/β) · g                            # guided marginal tilt
```

**No continuous diffusion, no flow-matching training, no BPTT.** Pure inference-time steering of discrete generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QGuidedDrafter (F1)                           │
│                                                                 │
│  SpeculativeGenerator  ──┬──> generate() (unguided, fallback)   │
│                          │                                      │
│  QGradientOracle ────────┤    ┌──────────────────────────┐      │
│  (LeoHead, FlowField,    │    │  project_one_step (F2)   │      │
│   ActionBridge, or       │───>│  one Euler-step lookahd  │──┐   │
│   BFN-rejection proxy)   │    └──────────────────────────┘  │   │
│                          │                                   ▼   │
│                          │    ┌──────────────────────────┐      │
│                          │    │  q_gradient_at (F3)      │      │
│                          │    │  ∇_a Q(s,â_1), J ≈ I     │      │
│                          │    └──────────────────────────┘      │
│                          │              │                      │
│                          │              ▼                      │
│                          │    guided_marginal = p_t + (1/β)·g  │
│                          │              │                      │
│                          └──────────────┴──────────────────────│
│                                                                 │
│  VarianceAdaptiveGuidance (F4): 1/β = sigmoid(k·(conf−τ))      │
└─────────────────────────────────────────────────────────────────┘
```

**Five-tier routing (Plasma/Hot/Warm/Cold/Freeze):**
- Plasma: ternary `ActionBridge` i8 directions + f32 Q dot product (< 100ns)
- Hot: cached `LeoHead::all_goals_q` f32 values (< 1μs)
- Warm: GPU batched Q-critic forward (~1ms)
- Cold: Turso-encrypted Q-table snapshots (~10ms load)
- Freeze: pure BC reference, no guidance (fallback)

---

## Tasks

### Phase 1: Core Primitives (unblock — no integration risk)

#### T1: QGradientOracle trait (F3)
- [x] Add `QGradientOracle` trait to `katgpt-core/src/traits.rs`
  ```rust
  pub trait QGradientOracle {
      type State;
      type Action;
      /// ∇_a Q(s, a) evaluated at the projected action.
      ///
      /// # QGF Design Decision (Research 268)
      /// Jacobian is intentionally dropped (J ≈ I) per QGF paper §5.
      /// Do NOT add chain-rule backprop — it increases variance (paper Fig 3).
      /// FFT smoothing (FlowFieldCache) is the equivalent variance reduction.
      fn q_gradient_at(&self, state: &Self::State, projected_action: &Self::Action) -> Vec<f32>;

      /// Zero-alloc variant — writes into caller-provided buffer.
      fn q_gradient_into(
          &self,
          state: &Self::State,
          projected_action: &Self::Action,
          out: &mut [f32],
      );

      /// Confidence in the gradient (for F4 adaptive weighting).
      /// Returns 1.0 for deterministic oracles, lower for noisy ones.
      fn confidence(&self, state: &Self::State) -> f32 { 1.0 }
  }
  ```
- [x] Unit test: mock oracle returns known gradient (via `NoGuidanceOracle` zero-test)
- [x] Unit test: `q_gradient_into` matches `q_gradient_at` for same input
- [x] Doc cross-ref to `.research/268_QGF_Test_Time_Q_Guided_Flow.md` §F3

#### T2: FirstOrderProjector (F2)
- [x] Create `katgpt-core/src/qgf/projector.rs`
- [x] Implement `project_one_step` for discrete chains
- [x] Implement batch variant `project_batch` using `generate_batch`
- [x] Unit test: known prefix → deterministic projection (mock generator)
- [x] Unit test: projection cost = 1 generator call (no BPTT)
- [ ] Benchmark: projection overhead < existing drafter call cost + 10%

#### T3: Feature gate scaffolding
- [x] Add `qgf` parent feature to `katgpt-core/Cargo.toml`
- [x] Add `qgf_projector`, `qgf_oracle`, `qgf_drafter`, `qgf_adaptive` sub-features
- [x] All OFF by default until GOAT proof
- [x] Wire `pub mod qgf;` in `katgpt-core/src/lib.rs` under `#[cfg(feature = "qgf")]`
- [x] Forward features from top-level `katgpt-rs/Cargo.toml`

---

### Phase 2: QGuidedDrafter (F1) — the core fusion

#### T4: QGuidedDrafter struct
- [ ] Create `katgpt-core/src/qgf/drafter.rs`
- [ ] Implement `QGuidedDrafter<G, O>` wrapping any `SpeculativeGenerator` + `QGradientOracle`
  ```rust
  pub struct QGuidedDrafter<G, O> {
      generator: G,
      oracle: O,
      guidance_weight: f32,  // 1/β
      guidance_period: usize, // apply guidance every N steps
  }
  ```
- [ ] Implement guided generation loop (discrete analogue of QGF Algorithm 1):
  ```rust
  // For each generation step t:
  //   1. Compute marginal p_t from generator
  //   2. Project prefix to final: â_1 = project_one_step(...)
  //   3. Query gradient: g = oracle.q_gradient_at(state, &â_1)
  //   4. Tilt marginal: p' = softmax(logits + (1/β) · g)  // sigmoid-gated
  //   5. Sample from p'
  ```
- [ ] Implement zero-alloc variant using pre-allocated marginal + gradient buffers
- [ ] Unit test: with `guidance_weight = 0.0`, QGuidedDrafter == base generator
- [ ] Unit test: with `guidance_weight > 0`, marginal is tilted toward higher-Q actions

#### T5: Implement QGradientOracle for existing types
- [ ] Impl `QGradientOracle` for `LeoHead` (delegate to `all_goals_q` + finite-difference)
- [ ] Impl `QGradientOracle` for `FlowFieldCache` (use `FlowField::gradient()` directly)
- [ ] Impl `QGradientOracle` for `ActionBridge<A, D>` (use i8 ternary directions as gradient proxy)
- [ ] Impl `QGradientOracle` for a new `BfnProxyOracle` (rejection-sampled return gradient — Freeze tier fallback)
- [ ] Unit test: each oracle returns sensible gradient for a known state

#### T6: NFCoT FlowScore fusion (unblock Plan 229)
- [ ] Extend `NfFlowScore` (Plan 229) to optionally consume Q-gradient guidance
- [ ] When `qgf_drafter` + `nf_flow_score` both enabled: QGF steers generation, NFCoT scores the result
- [ ] Test: QGF + NFCoT > NFCoT alone on Sudoku test suite (the unblock)
- [ ] Test: QGF + NFCoT > QGF alone (NFCoT adds ranking signal)
- [ ] Document the synergy in `.research/268` §8 (already done)

---

### Phase 3: VarianceAdaptiveGuidance (F4)

#### T7: Adaptive guidance weight
- [x] Create `katgpt-core/src/qgf/adaptive.rs`
- [x] Implement sigmoid-gated per-query guidance weight:
  ```rust
  /// guidance_weight = sigmoid(k · (confidence − threshold))
  /// - Low confidence → ~0 (pure BC reference, safe fallback)
  /// - High confidence → ~1 (strong Q-guidance)
  pub fn adaptive_guidance_weight(
      confidence: f32,
      threshold: f32,
      steepness: f32,
  ) -> f32;
  ```
- [x] Use **sigmoid, not softmax** (per project rules)
- [ ] Integrate with `QGuidedDrafter` — replaces fixed `guidance_weight` (needs T4)
- [ ] Reuse Thicket (Plan 267) variance probe as the confidence signal
- [x] Unit test: low confidence → weight ≈ 0; high confidence → weight ≈ 1
- [x] Unit test: threshold crossing is smooth (no discontinuity)
- [x] Unit test: monotonic in confidence, output range `[0,1]`

---

### Phase 4: Routing & Tier Integration

#### T8: CPU / SIMD / GPU / ANE auto-route
- [ ] Add `QgfComputeRoute` enum: `CpuSimd`, `GpuBatch`, `AneCritic`
- [ ] Implement `route_for(action_space_size, batch_size) -> QgfComputeRoute`:
  ```rust
  if action_space_size < 1024 { CpuSimd }
  else if batch_size >= 8 && action_space_size >= 1024 { GpuBatch }
  else { CpuSimd }
  ```
- [ ] Dispatch `q_gradient_at` to appropriate backend based on route
- [ ] CPU path: reuse existing `simd::dot_f32_i8` and `simd::fast_sigmoid`
- [ ] GPU path: batch dispatch via `riir-gpu` (optional, feature-gated)
- [ ] ANE path: route critic forward to `npc_ane_backend` (existing)
- [ ] Benchmark: routing decision is O(1) and does not dominate

#### T9: Plasma / Hot / Warm / Cold / Freeze tier wiring
- [ ] Document tier mapping in `qgf/mod.rs` (table from research doc §6)
- [ ] Plasma impl: `ActionBridge`-based ternary gradient (default for game NPCs)
- [ ] Hot impl: `LeoHead` cached Q-values (default for active inference)
- [ ] Warm impl: GPU batched critic (training-time / large batch)
- [ ] Cold impl: Turso Q-table loader (episode-end consolidation)
- [ ] Freeze impl: `NoGuidanceOracle` (returns zero gradient → pure BC reference)
- [ ] Test: Freeze tier produces identical output to unguided generator (graceful degradation)
- [ ] Test: tier promotion/demotion does not corrupt in-flight generation

---

### Phase 5: GOAT Proof — Before vs After

#### T10: GOAT benchmarks (the gate)
- [ ] Create `katgpt-core/benches/qgf_goat.rs` with feature-gated benchmarks
- [ ] **G1: First-attempt accuracy** — Sudoku 9×9 with vs without QGF
  - Baseline: DDTree + NFCoT FlowScore (Plan 229)
  - Target: +3-8% first-attempt solve rate
- [ ] **G2: Speculative acceptance rate** — DDTree spec bench
  - Baseline: DDTree greedy
  - Target: +5-12% acceptance
- [ ] **G3: Bomber arena win rate** — vs heuristic baseline
  - Baseline: current best
  - Target: +2-5% win rate
- [ ] **G4: Overhead** — prof_bench
  - Target: < 2% of total inference time
- [ ] **G5: Off-manifold false-positive** — OOD detection
  - Target: < 5% of guided actions are off-distribution

#### T11: Variance comparison (paper Fig 3 reproduction)
- [ ] Implement cosine-similarity variance test (paper Fig 3):
  - Compute `cos(G(s, a_t), G(s, a_t + ε))` for QGF, OOD, BPTT estimators
  - QGF should have highest cosine similarity (lowest variance)
- [ ] Note: we don't have a true BPTT path (intentionally not implemented),
  so compare QGF vs OOD vs identity-only
- [ ] Document result — validates the "drop Jacobian" decision

#### T12: Cross-feature integration tests
- [ ] QGF + NFCoT FlowScore (Plan 229) on Sudoku
- [ ] QGF + ThoughtFold (Plan 195) — guide, then fold, then re-guide
- [ ] QGF + ECHO (Plan 247) — ECHO provides the world model, QGF uses it as critic
- [ ] QGF + Thicket (Plan 267) — Thicket variance probe drives F4 adaptive weight
- [ ] Each test: enabled feature combo > baseline

---

### Phase 6: Documentation & Promotion

#### T13: Documentation
- [ ] Add QGF section to `katgpt-rs/README.md` Feature Showcase
- [ ] Update `katgpt-rs/.docs/01_overview.md` Feature Flags table
- [ ] Add `examples/qgf_01_guided_drafter.rs` — minimal usage
- [ ] Add `examples/qgf_02_adaptive_weight.rs` — F4 adaptive guidance
- [ ] Add `examples/qgf_03_tier_routing.rs` — plasma/hot/warm/cold/freeze demo
- [ ] Cross-link Research 268 ↔ Plan 268 ↔ Plan 229 (NFCoT)

#### T14: GOAT gate decision
- [ ] If G1-G5 all pass: promote `qgf_drafter` + `qgf_projector` + `qgf_oracle` to default-ON
- [ ] Keep `qgf_adaptive` (F4) opt-in until real-world validation on Bomber arena
- [ ] If QGF unblocks NFCoT (T6 passes strongly): promote `nf_flow_score` to default-ON too
- [ ] If any G fails: keep all QGF features opt-in, document the gap
- [ ] Update README with GOAT verdict

---

## Dependencies

### Existing (no new deps)
- `SpeculativeGenerator` trait (`katgpt-core/src/traits.rs`)
- `LeoHead` trait (`katgpt-core/src/traits.rs`, feature `leo_all_goals`)
- `FlowFieldCache` + `FlowField::gradient()` (`katgpt-core/src/flow/`)
- `ActionBridge` (`katgpt-core/src/bridge/`, feature `action_bridge`)
- NFCoT FlowScore (Plan 229, feature `nf_flow_score`)
- `simd::dot_f32_i8`, `simd::fast_sigmoid` (`katgpt-core/src/simd.rs`)
- `AutocurriculumSampler` (for BFN-proxy oracle, feature `dual_leo`)

### Optional (feature-gated)
- GPU dispatch via `riir-gpu` (for Warm tier, feature `qgf_gpu`)
- ANE critic forward via `npc_ane_backend` (existing)
- Turso Q-table loader (for Cold tier, feature `qgf_cold`)

---

## Expected GOAT Criteria (summary)

| Metric | Target | Gate |
|---|---|---|
| First-attempt accuracy (Sudoku) | +3-8% | G1 |
| Speculative acceptance rate | +5-12% | G2 |
| Bomber win rate | +2-5% | G3 |
| Guidance overhead | < 2% | G4 |
| Off-manifold false-positive | < 5% | G5 |
| Variance: QGF cos-sim > OOD cos-sim | ✅ | T11 |

**Promotion rule:** G1 ∧ G2 ∧ G4 ∧ G5 → promote F1+F2+F3 to default. G3 (Bomber) is a stretch goal. F4 stays opt-in.

---

## What This Plan Does NOT Do

- ❌ No continuous diffusion / flow-matching policy training (model-based, riir-ai)
- ❌ No IQL critic training (model-based, riir-ai — see companion doc)
- ❌ No BPTT gradient estimator (intentionally worse — documented in trait)
- ❌ No full-LLM training (LoRA-only per constraint 3, and that's riir-ai)
- ❌ No change to DDTree core (QGF wraps, does not modify)

---

## Open Risks

1. **Projection quality for discrete chains** — QGF's Euler step is continuous; our discrete analogue (single drafter call with collapsed budget) may not capture the same mode-selection benefit. T2 benchmark will reveal.
2. **NFCoT synergy may be weak** — if QGF alone > QGF+NFCoT, then NFCoT is redundant and should be deprecated. T6 will tell.
3. **Adaptive weight instability** — F4's sigmoid-gated weight could oscillate if confidence is noisy. Mitigation: EMA-smoothed confidence (reuse PrudentBanker pattern).
4. **Critic availability** — in Freeze tier, no critic → must fall back to BFN-proxy or pure BC. T9 Freeze impl handles this.

---

## References

- Paper: arXiv:2606.11087 (QGF, Zhou et al. 2026)
- Research: `.research/268_QGF_Test_Time_Q_Guided_Flow.md`
- Related plans: 229 (NFCoT FlowScore), 217 (NextLat), 247 (ECHO), 267 (Thicket), 195 (ThoughtFold)
- Related research: 150 (RecFM), 204 (NFCoT), 215 (ECHO), 229 (PAW), 267 (Thicket)
- riir-ai companion: `riir-ai/.research/123_QGF_Critic_Training_Verdict.md`
