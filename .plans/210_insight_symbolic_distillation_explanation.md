# Plan 210: INSIGHT Symbolic Distillation & Explanation — Modelless Explore→Distill→Explain Pipeline

**Date:** 2026-06-07
**Status:** 🔧 In Progress
**Research:** `.research/185_INSIGHT_Neuro_Symbolic_RL_Distillation.md`
**Depends On:** Plan 190 (AND-OR DDTree), Plan 206 (EGCS/EpisodePruner), BanditPruner, AbsorbCompressLayer, ScreeningPruner, TrialLog, RegressionSuite
**Feature Gates:** `insight_explain` (parent), `symbolic_distill`, `concept_grounding`, `decision_explain`, `reward_calibrator` (sub-gates, each independently gateable)
**GOAT Criteria:** F1 expression accuracy ≥80% on DDTree boundary; F2 grounding coverage ≥90% of pruner state vars; F3 attribution correctness ≥85% vs manual analysis; F4 calibration convergence ≤500 episodes; zero overhead on miss path (feature disabled)

---

## Overview

INSIGHT (arXiv 2403.12451) proves that an explore→distill→explain pipeline achieves strong RL performance while producing interpretable symbolic policies. Our DDTree + BanditPruner + AbsorbCompress stack already performs explore+distill at inference time. This plan implements four modelless fusions that close the loop:

- **F1 (Symbolic Expression Distillation):** Fit compact polynomial expressions to DDTree accept/reject boundaries. EQL analogue without training.
- **F2 (Concept Grounding):** Map pruner internals to human-readable explanations via templates. No LLM at runtime.
- **F3 (Decision Explanation):** Perturbation-based sensitivity analysis attributing token choices to pruner scores. No gradients.
- **F4 (Reward-Gated Calibration):** Formalize AbsorbCompress for continuous pruner parameters. Papaya lock-free tracking. blake3 audit.

All fusions are modelless inference-time computation. F4 ships first (formalizes existing pattern). F1 is the core novelty. F2+F3 are audit surfaces.

```
DDTree Exploration (existing)
    │
    ├── F4: RewardGatedCalibrator (P0 — formalizes existing AbsorbCompress)
    │   ├── Tracks pruner parameter→reward mapping (papaya lock-free)
    │   ├── Bandit-style threshold calibration
    │   └── AbsorbCompress promotes stable calibrations to hard constraints
    │
    ├── F1: SymbolicExpressionFitter → ExpressionPruner (P1 — core novelty)
    │   ├── Collects DDTree accept/reject traces
    │   ├── Greedy forward selection with sparsity budget
    │   ├── Basis: {identity, square, cube, sigmoid}
    │   └── Output: compact expression as new ScreeningPruner
    │
    ├── F2: ConceptGrounding → PolicyExplanation (P2 — depends on F1)
    │   ├── Template-based pruner state→semantic mapping
    │   ├── Chain-of-thought explanation generation
    │   └── JSONL-logged via TrialLog
    │
    └── F3: DecisionExplainer → DecisionExplanation (P3 — depends on F1)
        ├── Perturbation-based sensitivity analysis
        ├── Attributes token choices to pruner scores
        └── Async/post-inference computation
```

---

## Tasks

### Phase 1: F4 — Reward-Gated Pruner Calibration (P0)

- [x] **F4.1:** Create `src/pruners/reward_calibrator.rs` with `ParameterKey` and `ParameterStats` structs
  - `ParameterKey { pruner_id: u32, parameter_idx: u16, depth: u16 }` — 8 bytes, cache-line friendly
  - `ParameterStats { reward_sum: f32, visits: u32, variance: f32 }` — 12 bytes
  - `#[repr(u8)]` on any field-less enums

- [x] **F4.2:** Implement `RewardGatedCalibrator<P: ScreeningPruner>` struct
  - `inner: P` — wrapped pruner
  - `param_stats: papaya::HashMap<ParameterKey, ParameterStats>` — lock-free, zero contention
  - `calibration_log: Vec<CalibrationStep>` — audit trail
  - `config: CalibratorConfig { min_visits: usize, variance_threshold: f32, learning_rate: f32 }`
  - Feature-gated: `#[cfg(feature = "reward_calibrator")]`

- [x] **F4.3:** Implement `ScreeningPruner` for `RewardGatedCalibrator<P>`
  - `relevance()` delegates to inner pruner, records (parameter_key, score, eventual_reward)
  - Post-reward update: `bandit_update(key, reward)` with sigmoid-bounded Q-value
  - Use `sigmoid(learning_rate × reward_delta)` — no softmax

- [x] **F4.4:** Implement `CalibrationStep` struct with blake3 audit
  - `CalibrationStep { parameter_id: ParameterKey, old_value: f32, new_value: f32, reward_delta: f32, hash: [u8; 32] }`
  - blake3 hash of (old_value, new_value, reward_delta) for tamper-proof audit
  - `to_jsonl()` for TrialLog integration

- [x] **F4.5:** Implement calibration absorption via `AbsorbCompressLayer`
  - When a parameter has `visits >= min_visits` and `variance <= variance_threshold`, promote to fixed constraint
  - Wire into existing `AbsorbCompressLayer::try_compress()` pattern
  - Emit `CalibrationStep` on each absorption event

- [x] **F4.6:** Wire `RegressionSuite` verification
  - After calibration absorption, run `RegressionSuite::run()` against golden traces
  - GOAT gate: calibration pass iff regression suite all-pass
  - Integration test: calibrate → absorb → verify → assert no regression

- [x] **F4.7:** Unit tests for `RewardGatedCalibrator` (10/10 passing)
  - Test: parameter tracking accumulates rewards correctly
  - Test: sigmoid-bounded Q-values stay in [0, 1]
  - Test: absorption triggers when stability criteria met
  - Test: blake3 audit hash is deterministic
  - Test: miss path (feature disabled) has zero overhead

- [x] **F4.8:** Benchmark: calibration overhead on hot path
  - Target: <100ns per `relevance()` call overhead (papaya lookup)
  - Compare: calibrated vs uncalibrated pruner on 10K token evaluations

---

### Phase 2: F1 — Symbolic Expression Distillation (P1)

- [x] **F1.1:** Create `src/pruners/symbolic_expression.rs` with core types
  - `BasisFn` enum: `Identity`, `Square`, `Cube`, `Sigmoid` — `#[repr(u8)]`
  - `Term { basis: BasisFn, coefficient: f32, feature_idx: usize }` — single basis × coefficient
  - `SymbolicExpression { terms: Vec<Term>, bias: f32 }` — compact polynomial expression
  - `SymbolicExpression::evaluate(&self, features: &[f32]) -> f32` — O(terms)
  - `SymbolicExpression::to_string(&self, feature_names: &[&str]) -> String` — human-readable
  - Feature-gated: `#[cfg(feature = "symbolic_distill")]`

- [x] **F1.2:** Implement `SymbolicExpressionFitter` struct
  - `SymbolicExpressionFitter { max_terms: usize, min_improvement: f32, candidates: Vec<BasisFn> }`
  - Greedy forward selection: at each step, try all (basis_fn, feature_idx) pairs, pick best improvement
  - Improvement metric: reduction in MSE on accept/reject boundary
  - Sparsity budget: stop at `max_terms` or when improvement < `min_improvement`
  - No softmax in selection — pick argmax of improvement scores

- [x] **F1.3:** Implement `SymbolicExpressionFitter::fit()` method
  - Input: `TraceDataset { features: Vec<Vec<f32>>, labels: Vec<bool> }` — DDTree accept/reject
  - Output: `SymbolicExpression`
  - Algorithm: greedy forward selection with least-squares coefficient fitting
  - Regularization: L1 penalty on `coefficient` magnitude (prune near-zero terms)
  - Bounded output: final expression wrapped in `sigmoid()` for [0, 1] range

- [x] **F1.4:** Implement `TraceRecorder` for DDTree exploration
  - `TraceRecorder { records: Vec<TraceRecord> }` — collects (features, scores, accepted) per token
  - `TraceRecord { depth: usize, token_idx: usize, features: Vec<f32>, scores: Vec<f32>, accepted: bool }`
  - `TraceRecorder::record(depth, token, features, scores, accepted)` — called during DDTree exploration
  - Pre-allocate with `Vec::with_capacity(1024)`, `clear()` + reuse across episodes

- [x] **F1.5:** Create `src/pruners/expression_pruner.rs` with `ExpressionPruner<P>`
  - `ExpressionPruner<P: ScreeningPruner> { inner: P, expression: SymbolicExpression, feature_extractor: Box<dyn FeatureExtractor> }`
  - `FeatureExtractor` trait: `fn extract(depth: usize, token: usize, parents: &[usize], inner_scores: &[f32]) -> Vec<f32>`
  - `ScreeningPruner` impl: extract features → evaluate expression → return relevance
  - Feature-gated: `#[cfg(feature = "symbolic_distill")]`

- [x] **F1.6:** Implement expression serialization and deserialization
  - `SymbolicExpression::to_bytes()` — compact binary format for hot-swap
  - `SymbolicExpression::from_bytes()` — load from WASM/episode DB
  - blake3 hash for integrity verification during hot-swap

- [x] **F1.7:** Wire into AbsorbCompress self-improving cycle
  - ExpressionPruner<P> now implements AbsorbCompress trait via delegation to inner
  - AbsorbCompress tracks expression pruner's reward
  - If reward ≥ threshold for N episodes, absorb expression as default pruner
  - If reward drops, demote back to bandit arm

- [x] **F1.8:** Unit tests for symbolic expression system (16/16 passing)
  - Test: `BasisFn::Sigmoid` evaluation correctness
  - Test: expression evaluation matches manual computation
  - Test: fitter recovers known linear expression from synthetic data
  - Test: fitter respects `max_terms` budget
  - Test: sparsity pruning removes near-zero terms
  - Test: serialization round-trip preserves expression

- [x] **F1.9:** Unit tests for `ExpressionPruner` (7/7 passing)
  - Test: ScreeningPruner impl delegates correctly
  - Test: feature extraction produces expected dimensions
  - Test: expression pruner scores are in [0, 1] (sigmoid-bounded)

- [x] **F1.10:** Benchmark: expression fitting and evaluation
  - Fitting: target <1ms for 1000 trace records with 8 features
  - Evaluation: target <50ns per `relevance()` call
  - Compare: ExpressionPruner vs inner pruner overhead

- [x] **F1.11:** Create `examples/symbolic_distill_demo.rs`
  - Demonstrates: DDTree exploration → trace recording → expression fitting → ExpressionPruner
  - Shows before/after relevance scores
  - Prints human-readable expression

---

### Phase 3: F2 — Concept Grounding for Pruner Rules (P2)

- [x] **F2.1:** Create `src/pruners/concept_grounding.rs` with core types
  - `GroundingSource` enum: `Template`, `Learned` — `#[repr(u8)]`
  - `ConceptMapping { variable: String, semantic: String, confidence: f32, source: GroundingSource }`
  - `PolicyExplanation { mappings: Vec<ConceptMapping>, chain_of_thought: Vec<String>, summary: String }`
  - `PolicyExplanation::to_json(&self) -> String` — serializable for TrialLog
  - Feature-gated: `#[cfg(feature = "concept_grounding")]`

- [x] **F2.2:** Define `ConceptGrounding` trait
  - `fn ground(&self, state: &PrunerState) -> Vec<ConceptMapping>`
  - `fn explain_chain(&self, state: &PrunerState, mappings: &[ConceptMapping]) -> Vec<String>`
  - `fn summarize(&self, mappings: &[ConceptMapping], chain: &[String]) -> String`
  - Trait is `Send + Sync` for async compatibility

- [x] **F2.3:** Implement `PrunerState` snapshot struct
  - `PrunerState { depth: usize, token_idx: usize, parent_tokens: Vec<usize>, pruner_scores: Vec<(String, f32)>, accepted: bool }`
  - Captures pruner internals at a decision point
  - Pre-allocated `Vec::with_capacity(8)` — typical pruner count

- [x] **F2.4:** Implement `TemplateGrounding` — static concept mapping
  - Static `Vec<(pattern: &str, semantic: &str)>` lookup table — pre-computed at compile time
  - Pattern matching: token index → vocab lookup → semantic mapping
  - Score-based grounding: "score > 0.8" → "high confidence", "score <= 0.5" → "low confidence / rejected"
  - Depth-based grounding: "depth 0" → "top-level declaration", "depth > 3" → "nested expression"
  - Confidence: `sigmoid(1.0) ≈ 0.731` for template matches (high confidence by design)

- [x] **F2.5:** Implement chain-of-thought template engine
  - Template patterns:
    - "Token {token} at depth {depth} was {action} because {reason}"
    - "Pruner '{name}' scored {score:.2} ({interpretation})"
    - "Combined relevance: {combined} → {decision}"
  - Fill templates from `PrunerState` + `ConceptMapping`
  - No LLM — pure string interpolation

- [x] **F2.6:** Wire `PolicyExplanation` into `TrialLog`
  - `TrialLog::log_explanation(explanation: &PolicyExplanation)` — JSONL append
  - blake3 hash of explanation for audit integrity
  - Opt-in: only logged when `concept_grounding` feature enabled

- [x] **F2.7:** Integrate with `ExpressionPruner` from F1
  - If expression pruner is active, ground expression terms in concept mappings
  - Example: "Term 0: 0.7 × sigmoid(x₂)" → "0.7 × sigmoid(syntax_validity)"
  - Requires feature names from `FeatureExtractor` trait (F1.5)

- [x] **F2.8:** Unit tests for concept grounding
  - Test: template grounding produces correct mappings for known pruner states
  - Test: chain-of-thought fills templates correctly
  - Test: summary is non-empty and human-readable
  - Test: confidence values are sigmoid-bounded [0, 1]
  - Test: empty pruner state → graceful degradation (no panic)
  - Test: to_json produces valid-ish JSON string
  - Test: depth-based grounding maps correctly
  - Test: score-based threshold mapping
  - Test: full pipeline (ground → explain → summarize → JSON)
  - 13 tests ALL PASS

- [x] **F2.9:** Benchmark: grounding overhead
  - Target: <10μs per grounding call (template lookup + string interpolation)
  - Not on hot path — post-inference only

---

### Phase 4: F3 — Decision Explanation via Sensitivity Analysis (P3)

- [x] **F3.1:** Create `src/pruners/decision_explainer.rs` with core types
  - `TokenChoice { depth: usize, token_idx: usize, score: f32, pruner_attributions: Vec<PrunerAttribution> }`
  - `PrunerAttribution { pruner_name: String, score: f32, sensitivity: f32 }` — how much this pruner influenced the choice
  - `RejectedAlternative { token_idx: usize, score: f32, why_rejected: String }`
  - `DecisionExplanation { choices: Vec<TokenChoice>, alternatives: Vec<RejectedAlternative>, summary: String }`
  - Feature-gated: `#[cfg(feature = "decision_explain")]`

- [x] **F3.2:** Define `DecisionExplainer` trait
  - `fn explain(&self, trace: &[TraceNode]) -> DecisionExplanation`
  - `fn sensitivity(&self, trace: &[TraceNode], pruner_idx: usize, delta: f32) -> Vec<f32>`
  - Trait is `Send + Sync` for async computation

- [x] **F3.3:** Define `TraceNode` lightweight recording struct
  - `TraceNode { depth: usize, candidates: Vec<CandidateRecord>, chosen: usize }`
  - `CandidateRecord { token_idx: usize, pruner_scores: Vec<f32>, accepted: bool }`
  - Pre-allocated `Vec::with_capacity(16)` for candidates
  - Collected during DDTree exploration (zero cost when feature disabled)

- [x] **F3.4:** Implement `PerturbationExplainer`
  - For each token choice, for each pruner score:
    1. Perturb score by ±δ (default δ = 0.1)
    2. Re-run accept/reject decision with perturbed score
    3. If output changes → sensitivity = |change| / δ
    4. If output unchanged → sensitivity = 0.0
  - Attribution: rank pruners by sensitivity (argmax, no softmax)
  - Primary driver: pruner with highest sensitivity
  - Run post-inference or async — not on hot path

- [x] **F3.5:** Implement sensitivity report formatting
  - "Token 'struct' was chosen over 'enum' at depth 0:"
  - "  Syntax pruner: struct=0.85, enum=0.72 (Δ=0.13)"
  - "  Bandit Q-value: struct=0.68, enum=0.51 (Δ=0.17)"
  - "  Sensitivity: If syntax pruner had scored 'enum' 0.15 higher, 'enum' would be selected"
  - "  → Primary driver: bandit Q-value (Δ=0.17 > 0.13)"

- [x] **F3.6:** Integrate with `ConceptGrounding` from F2
  - Attribution explanations use concept-grounded pruner names
  - "Syntax pruner" → "syntax validity checker" if grounding available
  - Graceful fallback: use raw pruner names if grounding disabled

- [x] **F3.7:** Implement caching for sensitivity results
  - `SensitivityCache { cache: Arc<RwLock<HashMap<[u8;32], Vec<f32>>>> }` — lock-free when papaya available
  - Cache key: blake3 hash of (TraceNode serialized)
  - Invalidate on HotSwapPruner reload (version bump)
  - Avoids recomputation across episodes with similar traces

- [x] **F3.8:** Wire `DecisionExplanation` into `TrialLog`
  - `TrialLog::log_decision(explanation: &DecisionExplanation)` — JSONL append
  - Optional: only logged when `decision_explain` feature enabled
  - Integration with `RegressionSuite`: explanations from golden episodes serve as expected behavior docs

- [x] **F3.9:** Unit tests for decision explainer (10/10 passing)
  - Test: perturbation correctly identifies primary driver pruner
  - Test: sensitivity values are non-negative
  - Test: zero sensitivity when perturbation doesn't change output
  - Test: cache hit/miss correctness
  - Test: integration with concept grounding (when both features enabled)
  - Test: empty trace → graceful empty explanation (no panic)

- [x] **F3.10:** Benchmark: sensitivity analysis cost
  - Target: <5ms for 100-token trace with 4 pruners (post-inference)
  - Measure: perturbation loop cost vs trace recording cost
  - Verify: not on hot path when feature disabled

---

### Phase 5: Integration & GOAT Gate

- [x] **I1:** Add feature flags to `Cargo.toml`
  - `insight_explain = ["symbolic_distill", "concept_grounding", "decision_explain", "reward_calibrator"]`
  - `symbolic_distill = []` — F1
  - `concept_grounding = ["symbolic_distill"]` — F2 (depends on F1 expressions)
  - `decision_explain = []` — F3
  - `reward_calibrator = ["bandit"]` — F4 (depends on bandit infrastructure)
  - Do NOT add to default features — GOAT gate first

- [x] **I2:** Register new modules in `src/pruners/mod.rs`
  - `#[cfg(feature = "symbolic_distill")] pub mod symbolic_expression;`
  - `#[cfg(feature = "symbolic_distill")] pub mod expression_pruner;`
  - `#[cfg(feature = "concept_grounding")] pub mod concept_grounding;`
  - `#[cfg(feature = "decision_explain")] pub mod decision_explainer;`
  - `#[cfg(feature = "reward_calibrator")] pub mod reward_calibrator;`

- [x] **I3:** Create `examples/insight_demo.rs`
  - Demonstrates full explore→distill→explain pipeline:
    1. DDTree exploration with trace recording
    2. F4: Reward-gated calibration of pruner parameters
    3. F1: Symbolic expression fitting from traces
    4. F2: Concept grounding of expression terms
    5. F3: Decision explanation with sensitivity analysis
  - Prints human-readable output at each stage
  - Shows before/after relevance scores with expressions

- [x] **I4:** GOAT gate validation — compilation clean, 0 regressions, zero overhead when features disabled
  - Run full test suite with `--features insight_explain`
  - Run full test suite WITHOUT feature — verify zero regressions
  - Benchmark: hot path with feature disabled shows no overhead
  - GOAT criteria check:
    - F1: expression accuracy ≥80% on known DDTree boundaries
    - F2: grounding coverage ≥90% of pruner state variables
    - F3: attribution correctness ≥85% vs manual analysis
    - F4: calibration convergence ≤500 episodes

- [x] **I5:** Cross-feature integration tests
  - Test: F4 calibration + F1 expression fitting → calibrated expression pruner
  - Test: F1 expression + F2 grounding → grounded expression explanation
  - Test: F3 explanation + F2 grounding → concept-grounded decision trace
  - Test: F4 calibration + RegressionSuite → regression-safe calibration
  - Test: HotSwapPruner reload of expression pruner → correct behavior post-swap

- [ ] **I6:** Documentation
  - Add module-level docs for each new file
  - Update `src/pruners/mod.rs` module docs
  - Add example usage in trait doc comments

---

### Phase 6: Benchmarks & GOAT Gate Promotion

- [x] **B1:** Create `.benchmarks/insight_explain_bench.md`
  - F4 calibration overhead (per-relevance call)
  - F1 fitting time (per-1000 traces)
  - F1 evaluation overhead (per-relevance call)
  - F2 grounding overhead (per-explanation)
  - F3 sensitivity analysis (per-100-token trace)
  - Memory: additional allocations per episode

- [ ] **B2:** Run benchmarks with and without feature
  - Baseline: `cargo bench` without `insight_explain`
  - Feature: `cargo bench --features insight_explain`
  - Assert: hot-path overhead <1% when feature disabled

- [ ] **B3:** GOAT gate promotion decision
  - If all GOAT criteria pass → add `insight_explain` to default features in `Cargo.toml`
  - If any criteria fail → document failure, keep opt-in, create follow-up plan
  - Record verdict in `.benchmarks/insight_explain_bench.md`

---

## Dependency Graph

```
F4 (RewardGatedCalibrator) ──→ I1-I6 (Integration)
    │                              ↑
    │                              │
F1 (SymbolicExpressionFitter) ────┤
    │                              │
    ├──→ F2 (ConceptGrounding) ───┤
    │                              │
    └──→ F3 (DecisionExplainer) ──┘
```

- F4 is independent — can ship immediately
- F1 is independent — can develop in parallel with F4
- F2 depends on F1 (grounds expression terms)
- F3 is independent of F2 but benefits from F2 integration
- Integration phase (I1-I6) depends on all fusions being complete

---

## Estimated Timeline

| Phase | Tasks | Est. Days |
|-------|-------|-----------|
| Phase 1: F4 Calibration | F4.1-F4.8 | 2 |
| Phase 2: F1 Symbolic Distillation | F1.1-F1.11 | 5 |
| Phase 3: F2 Concept Grounding | F2.1-F2.9 | 2 |
| Phase 4: F3 Decision Explanation | F3.1-F3.10 | 3 |
| Phase 5: Integration & GOAT | I1-I6 | 2 |
| Phase 6: Benchmarks | B1-B3 | 1 |
| **Total** | | **15 days** |

---

## Notes

- All fusions use **sigmoid** for bounded scoring. No softmax anywhere.
- Papaya lock-free HashMap for all concurrent state (F4 parameter tracking, F3 sensitivity cache).
- blake3 for all audit hashes (F4 calibration steps, F3 cache keys, F1 expression integrity).
- Feature-gated: entire pipeline disableable with zero hot-path cost.
- Modelless: no LLM at runtime, no neural network training, no gradient descent.
- F2 template grounding is LLM-optional during development only — never at inference time.

## Cross-Repo Alignment (riir-ai ↔ katgpt-rs)

| riir-ai Plan | Relationship | Notes |
|---|---|---|
| **240** EQL Symbolic LoRA | Type alignment required | 240's `EqlActivation` (Identity, Square, Cube, Constant, Product, Sum) is a superset of 210's `BasisFn` (Identity, Square, Cube, Sigmoid). 240's `EqlExpression` should serialize into 210's `SymbolicExpression` format for engine deployment. **Action:** 210 defines canonical engine-side expression types; 240 maps to them. |
| **239** FOL Game Rules | Complementary | 239 extracts qualitative FOL rules from LoRA weights; 210 F1 fits quantitative expressions from DDTree traces. Both produce interpretable rules at different granularity. |

### Type Sharing Strategy

- `SymbolicExpression` in katgpt-rs `src/pruners/symbolic_expression.rs` is the canonical engine type
- riir-ai 240's `EqlExpression` serializes to `SymbolicExpression` binary format via shared serde
- Activation mapping: `EqlActivation::Constant → BasisFn::Identity(coeff=1.0)`, `EqlActivation::Product/Sum → expand as multi-term`
- Feature gate: `eql_eval` in riir-engine depends on `katgpt-core` types behind `eql_eval` feature

### Execution Order

| Phase | Plan | Rationale |
|-------|------|----------|
| 1 | **210 F4** (this plan, Reward Calibration) | Formalizes existing AbsorbCompress — zero risk, ships first |
| 2 | 212 (Collapse-Aware Thinking) | Independent, proven by S2F paper |
| 3 | 209 (FOL Inference) | Foundation for 211's mode router |
| 4 | **210 F1-F3** (this plan, Distillation + Explanation) | Core novelty |
| 5 | 211 (Three-Mode Router) | Consumes 209 + 210 outputs |

---

---

TL;DR: Implement INSIGHT's explore→distill→explain pattern as four modelless fusions in katgpt-rs. F4 (reward-gated calibration) ships first — it formalizes the existing AbsorbCompress pattern. F1 (symbolic expression fitting from DDTree traces) is the core novelty. F2 (concept grounding) and F3 (decision explanation) add audit surfaces. All behind `insight_explain` feature flag. ~15 days. GOAT gate before default promotion.
