# Plan 162: Emotion Vector Inference-Time Behavior Control

## Context

**Research:** 144 (Functional Emotions — Linear Representations for Behavior Control)
**Source:** Anthropic Transformer Circuits Thread, 2026 — emotion vectors causally drive behavior (desperation → 14× reward hacking increase)
**Verdict:** GAIN — zero-cost modelless observation, default-on if GOAT proof passes

## Summary

Read emotion vector projections (desperation, calm, valence PC1) from mid-layer residual stream activations during speculative decoding. Use as:
1. **Desperation monitor** — early warning for reward-hacking-prone regimes in DDTree
2. **Valence supplement** to `ReviewMetrics` — richer than entropy anomaly alone
3. **SR²AM feature** — feed into configurator bandit context for adaptive planning

Zero extra forward pass. O(d) dot product per decode step. Already computed by model.

## Tasks

### Phase 1: Infrastructure (Modelless Read)

- [x] T0: Plan creation
- [x] T1: Add `EmotionDirections` struct with pre-computed direction vectors (`desperation`, `calm`, `valence`, `arousal`) loaded from model config — `src/pruners/emotion_vector.rs`
- [x] T2: Add `emotion_valence_sum`, `emotion_arousal_sum`, `desperation_score_sum`, `calm_score_sum`, `emotion_count` atomic fields to `ReviewMetrics` — `src/pruners/review_metrics.rs`
- [x] T3: `EmotionDirections::project(activation, direction) -> f32` — zero-alloc O(d) dot product — `src/pruners/emotion_vector.rs`
- [x] T4: `EmotionDirections::read_emotions(activations) -> EmotionReading` + `ReviewMetrics::record_emotion()` — `src/pruners/emotion_vector.rs` + `src/pruners/review_metrics.rs`
- [x] T5: `is_desperate_session(threshold)` + `emotion_profile_summary()` on `ReviewMetrics` — `src/pruners/review_metrics.rs`
- [x] T6: `ReviewMetrics::Display` updated with emotion profile — `src/pruners/review_metrics.rs`

### Phase 2: GOAT Proof (Benchmark)

- [ ] T7: Create GOAT proof — benchmark decode throughput with/without emotion vector reading
  - Must show: zero measurable overhead (< 0.1% decode time increase)
  - Test on same-commit, back-to-back runs per optimization.md
- [ ] T8: Create GOAT proof — measure correlation between desperation_score and existing entropy_anomaly
  - Must show: desperation_score provides strictly more information than entropy alone (non-trivial correlation but not perfect correlation)
- [ ] T9: Run clippy, fix warnings

### Phase 3: Integration (Default-On if GOAT Passes)

- [ ] T10: If T7 passes (no perf hurt), make emotion vector reading default-on (no feature gate needed)
- [ ] T11: If T8 passes (information gain), integrate desperation_score into `SR2AMConfig` context features
- [ ] T12: Add `emotion_desperation_threshold` to domain config (with sensible default)
- [ ] T13: Update README with emotion vector monitoring section

## Architecture

### New Types

```rust
/// Pre-computed emotion direction vectors loaded from model config.
/// Fixed per model — compute once at load time, zero-alloc reads during decode.
pub struct EmotionDirections {
    /// Valence PC1 direction (positive = happy/calm, negative = desperate/angry)
    pub valence: Vec<f32>,    // [d_model]
    /// Arousal PC2 direction (positive = high arousal, negative = low arousal)
    pub arousal: Vec<f32>,    // [d_model]
    /// Desperation-specific direction
    pub desperation: Vec<f32>, // [d_model]
    /// Calm-specific direction
    pub calm: Vec<f32>,       // [d_model]
}

impl EmotionDirections {
    /// Project activation vector onto a direction — O(d) dot product, zero alloc
    pub fn project(activation: &[f32], direction: &[f32]) -> f32 {
        activation.iter()
            .zip(direction.iter())
            .map(|(a, d)| a * d)
            .sum()
    }
}
```

### Integration Points

| File | Change |
|------|--------|
| `src/pruners/review_metrics.rs` | Add emotion fields, project methods, summaries |
| `src/pruners/configurator_bandit.rs` | Add desperation/calm to context features |
| `src/transformer.rs` | Hook emotion projection at mid-layer during decode |
| `src/types.rs` | Add `EmotionDirections` to config |

### Feature Gate

**None by default** — if GOAT proof shows zero overhead, emotion reading is always on.
If overhead is measurable (>0.1%), gate behind `emotion_vector_read` feature flag.

### Data Flow

```
Decode Step
  ├─ Mid-layer activations computed (already happens)
  ├─ project(activations, desperation_dir) → f32
  ├─ project(activations, calm_dir) → f32
  ├─ project(activations, valence_dir) → f32
  ├─ Update ReviewMetrics (pre-allocated fields)
  └─ If desperation > threshold → flag for SR²AM configurator
```

### Optimization Compliance (from optimization.md)

- Pre-compute directions once in config struct (Do: Data Structures)
- Store projections in pre-allocated ReviewMetrics fields (Do: Allocation)
- O(1) dot product, no linear scan (Do: Caching)
- No Rayon — single dot product is ~0.01μs (Don't: Rayon for tiny workloads)
- No allocation in decode loop (Don't: Allocate inside hot loops)
- Compare same-commit, back-to-back runs (Do: Profiling)

## GOAT Proof Criteria

| Criterion | Threshold | Pass Condition |
|-----------|-----------|----------------|
| Decode throughput | < 0.1% regression | Emotion read adds zero measurable cost |
| Binary size | < 1% increase | New code is minimal |
| Information gain | r < 0.9 with entropy | Desperation not redundant with entropy |
| Desperation-reward correlation | r > 0.3 | Desperation score predicts failure regimes |

**All 4 must pass for default-on.** If any fails, feature-gate and revisit.

## Dependencies

- Plan 061 (Entropy Anomaly) — extends ReviewMetrics
- Plan 112 (SR²AM Configurator Bandit) — integration target
- Research 037 (Model-Based/Modelless Duality) — architectural framework
