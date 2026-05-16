# Plan 061: Entropy Anomaly Detection — Session-Level OOD Monitoring

## Tasks

- [x] T0: Plan creation
- [x] T1: Add `mean_prediction_error()` to `DeltaMemoryState`
- [x] T2: Add `EntropyAnomaly` fields to `ReviewMetrics` (entropy sum, count, max spike)
- [x] T3: Add `entropy_anomaly_summary()` and `is_high_entropy_session()` to `ReviewMetrics`
- [x] T4: Update `ReviewSummary` and `Display` impl with entropy anomaly
- [x] T5: Add tests for new `ReviewMetrics` entropy tracking
- [x] T6: Add tests for `DeltaMemoryState::mean_prediction_error()`
- [x] T7: Run clippy, fix warnings, commit

## Architecture

### What This Actually Is

Session-level Out-Of-Distribution (OOD) detection using signals that already exist in the codebase:

1. **PPoT entropy** — `token_entropy()` already computes Shannon entropy per position. If a user's session consistently pushes entropy above baseline, the model is confused by their inputs. This is a real, measurable anomaly signal.

2. **DeltaMemory prediction error** — `DeltaMemoryState` already tracks `error_history` per write. Exposing `mean_prediction_error()` gives a scalar drift signal: if prediction errors spike, current inputs don't match learned patterns.

### What This Is NOT

- NOT a Mahalanobis distance monitor (DeltaMemoryState is associative KV memory, not a covariance matrix)
- NOT a WASM rejection rate monitor (bomber validator rejects game moves, not user inputs)
- NOT a separate `SuspicionMetrics` struct (extends existing `ReviewMetrics` instead)

### File Changes

| File | Change |
|------|--------|
| `src/pruners/delta_mem/state.rs` | Add `mean_prediction_error()` method |
| `src/pruners/review_metrics.rs` | Add entropy anomaly fields + methods + Display |
| `tests/` | New test for entropy anomaly tracking |

### Signal Design

#### Entropy Anomaly (from PPoT)

```text
record_entropy(h: f32) → updates running sum/count/max
entropy_anomaly_summary() → EntropyAnomalySummary { mean, max, count }
is_high_entropy_session(threshold) → bool
```

Threshold: `ln(vocab_size) * 0.7` as default (70% of max entropy = model is quite uncertain).
For micro config (vocab=32): threshold ≈ `ln(32) * 0.7 ≈ 2.42`.

#### Delta Memory Prediction Error (from DeltaMemoryState)

```text
mean_prediction_error() → f32 (0.0 if no history)
```

Already tracked internally as `error_history`. Just expose the mean.

### Why Extend ReviewMetrics (Not New Struct)

1. Already thread-safe (`AtomicU64`)
2. Already wired into `BanditSession` via `with_metrics()`
3. Already displayed in examples (`review_01_metrics.rs`)
4. Central telemetry point — adding entropy here keeps all session metrics in one place

## Expected Outcomes

### Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| `mean_prediction_error()` returns correct mean | Within 1e-6 of manual calculation |
| `entropy_anomaly_summary()` accumulates correctly | Sum/count/max match individual recordings |
| `is_high_entropy_session()` triggers above threshold | Mean > threshold → true |
| No new dependencies | Uses existing `AtomicU64` |
| All existing tests pass | No regressions |

### What This Proves

- ✅ Whether entropy monitoring is useful for session-level anomaly detection
- ✅ Whether DeltaMemory prediction error is a viable drift signal
- ✅ Minimal infrastructure for future OOD work

### What This Does NOT Prove

- ❌ Whether these signals actually catch malicious users (needs real data)
- ❌ Whether a combined "suspicion score" is better than individual signals
- ❌ Whether these thresholds work in production (needs calibration)

## Key Design Decisions

1. **AtomicU64 for entropy sum** — Store `entropy * 10000.0` as u64 (same pattern as `path_consistency_sum`). 4 decimal places of precision, zero lock overhead.

2. **No separate anomaly module** — Too small to justify a new file. Entropy tracking is ~30 lines in `ReviewMetrics`.

3. **Expose `mean_prediction_error()` on DeltaMemoryState** — The error history is already maintained. Just compute the mean. No new storage.

4. **Default threshold from vocabulary size** — Makes the threshold config-independent. User can override.

5. **Feature gate: `bandit`** — Same gate as existing `ReviewMetrics`. No new feature flags.

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Entropy signals too noisy | Medium | Threshold is configurable; start conservative |
| `mean_prediction_error()` slow for large history | Low | error_history is bounded by `error_window * rank` |
| Over-interpretation of anomaly scores | Medium | Documentation clearly states this is exploratory |