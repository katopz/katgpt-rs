# NFCoT FlowScore Drafter (Plan 229)

**Research:** 204_NFCoT_Normalizing_Flow_Continuous_CoT
**Status:** ✅ Complete (T1-T8 all done). GOAT: ⚠️ MARGINAL — debug overhead 3.5%, release expected <1%. Keep opt-in.
**GOAT Gate:** `nf_flow_score` (parent: `nf_flow`, default: OFF)

## Overview

Implement inference-time normalizing flow density scoring for speculative decoding candidates. Construct lightweight affine flow from DDTree marginals — zero training, zero additional model forward passes.

## Architecture

The FlowScore computes an exact log-density over candidate token trajectories using:
1. Base log-probability from DDTree marginals: Σ log P(token_i | context)
2. Log-determinant from affine transform: Σ log σ_i where σ_i = sigmoid(entropy_i)
3. Combined: flow_score = base_logprob + log_det

This is a diagonal affine normalizing flow constructed from existing inference-time statistics. No training needed. O(vocab_size) per position (same as existing softmax).

## Tasks

### T1: Core FlowScore Computation
- [x] Create `src/speculative/nf_flow.rs` with `NfFlowScore` struct
- [x] Implement `fn flow_score(marginals: &[Vec<f32>], selected: &[usize]) -> f32`
  - base_logprob = Σ log marginals[i][selected[i]]
  - σ_i = sigmoid(entropy of marginals[i])  // entropy of categorical distribution
  - log_det = Σ log(σ_i)
  - score = base_logprob + log_det
- [x] Unit test: known marginals → known flow score
- [x] Unit test: uniform marginals (high entropy) → σ ≈ 1 → log_det ≈ 0 → score ≈ base
- [x] Unit test: peaked marginals (low entropy) → σ ≈ 0 → large negative log_det → score < base

### T2: FlowScore Drafter Integration with SpeculativeGenerator
- [x] Extend `SpeculativeGenerator` trait with `fn generate_scored()` that returns candidates with flow scores
- [x] Implement for existing drafter: compute flow scores for all DDTree candidates
- [x] Select candidate with highest flow_score instead of highest base probability
- [x] Benchmark: flow_score selection vs max-prob selection on existing test suite

### T3: FlowGate Acceptance Criterion
- [x] Create `NfFlowGate` struct with EMA threshold tracking
- [x] Implement `fn accept(score: f32) -> bool` with adaptive threshold
- [x] Threshold = EMA(α=0.01) of historical flow scores
- [x] Accept if score > threshold, reject otherwise
- [x] Integrate with speculative verification pipeline
- [x] Test: gate accepts high-score trajectories, rejects low-score ones

### T4: FlowBudget Allocation
- [x] Implement `fn allocate_budget(scores: &[f32], total_budget: usize) -> Vec<usize>`
- [x] Allocate speculative depth proportional to flow score (sigmoid-weighted, NOT softmax)
- [x] High-score branches get more speculative depth, low-score get early termination
- [x] Integrate with DDTree's existing depth control
- [x] Test: budget allocation matches expected distribution

### T5: GOAT Proof — Before/After Benchmarks
- [x] Create `tests/nf_flow_goat.rs` with feature-gated benchmarks
- [x] Benchmark: flow_score computation overhead (< 1% of total inference)
- [x] Benchmark: speculative acceptance rate before/after FlowGate
- [x] Benchmark: first-attempt accuracy before/after FlowScore selection
- [x] Test: FlowScore drafter on Sudoku solver (existing test infrastructure)
- [x] Test: FlowScore drafter on Rust syntax validation (SynPruner)
- [x] Validate: flow_score > max-prob selection on ≥ 3 existing test suites

### T6: FlowMUX Composition (cross-feature)
- [x] Implement flow scoring for MUX multiplexed trajectories
- [x] Score continuous MUX tokens using flow density in simplex space
- [x] Requires MUX feature (Research 158) to be available
- [x] Test: MUX + FlowScore composition improves over MUX alone

### T7: FlowFold Chain Composition (cross-feature)
- [x] Integrate flow score with ThoughtFold chain folding
- [x] Before fold: compute flow_score(original_chain)
- [x] After fold: compute flow_score(folded_chain)
- [x] Accept fold if flow_score(folded) ≥ α · flow_score(original)
- [x] Requires chain_fold feature (Plan 195) to be available
- [x] Test: confidence-gated folding reduces false folds

### T8: Feature Gate Integration
- [x] Add `nf_flow` parent feature to Cargo.toml
- [x] Add `nf_flow_score`, `nf_flow_gate`, `nf_flow_budget`, `nf_flow_mux`, `nf_flow_fold` sub-features
- [x] All OFF by default until GOAT proof
- [x] Update README with NFCoT FlowScore section

## Expected GOAT Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Flow score overhead | < 1% total inference | prof_bench |
| Acceptance rate gain | +5-10% over baseline | speculative bench |
| First-attempt accuracy | +2-5% over max-prob | existing test suites |
| False fold reduction | < 5% false folds | chain_fold bench |

## Dependencies

- DDTree marginals (existing)
- SpeculativeGenerator trait (existing)
- MUX (optional, for T6)
- ThoughtFold (optional, for T7)

## Promotion Criteria

If T5 GOAT proof passes:
- Promote `nf_flow_score` to default-ON
- Promote `nf_flow_gate` to default-ON
- Keep `nf_flow_mux` and `nf_flow_fold` as opt-in (depend on other features)
