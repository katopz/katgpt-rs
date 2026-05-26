# Plan 136: Training-Free Loop Wrapper (ODE-Refined Sub-Stepping)

> **Research:** [097 — Training-Free Looped Transformers](../.research/097_Training_Free_Looped_Transformers.md)
> **Paper:** [arXiv:2605.23872](https://arxiv.org/abs/2605.23872) — Training-free mid-block looping with damped Euler sub-stepping
> **Feature Gate:** `tf_loop` (opt-in, depends on `lt2_looped`)
> **Status:** ✅ Complete (Phase 0–5 ✓, GOAT 4/4 passing)

## Summary

Add a training-free loop wrapper to katgpt-rs that re-applies a contiguous mid-stack block of layers with ODE-motivated damped sub-stepping. Unlike our existing LT2 (Plan 108, training-time weight-sharing), this requires **zero training** — it's a pure inference-time retrofit on frozen checkpoints.

The key insight from the paper: each pre-norm transformer layer is a forward Euler step at h=1 on a residual ODE. Naive looping advances to t=K (catastrophic). Damped sub-stepping at h=1/K stays at t=1 but with better approximation. The K-stage Runge-Kutta with β anchor is the only robust strategy — all higher-order methods (Anderson, RK4, heavy-ball) fail.

**Why this matters for us:** Our LT2 does whole-model weight-shared looping. This paper shows that (a) only a 4-layer mid-stack window matters, (b) sub-stepping is essential, and (c) layer-mode is required for MoE. We can adopt all three insights.

---

## Tasks

### Phase 0: Core Types (katgpt-core)
- [x] T0: Add `SubStepStrategy` enum to `katgpt-core/src/types.rs`
  ```rust
  /// Sub-stepping strategy for training-free loop refinement.
  #[derive(Clone, Copy, Debug, Default, PartialEq)]
  pub enum SubStepStrategy {
      /// Damped Euler: x_{k+1} = (1-1/K)·x_k + (1/K)·g(x_k)
      #[default]
      DampedEuler,
      /// K-stage Runge-Kutta with anchor β.
      /// x₁ = β·g(x₀) + (1-β)·F^K(x₀)
      KStageRK { beta: f32 },  // default β=0.5
  }
  ```
- [ ] T1: Add `IterationMode` enum (block vs layer)
  ```rust
  /// How loop iterations are applied to the window.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub enum IterationMode {
      /// Block-mode: iterate entire window as one unit. Default for dense.
      #[default]
      Block,
      /// Layer-mode: iterate each layer K times before moving on. Required for MoE.
      Layer,
  }
  ```
- [x] T2: Add `CacheStrategy` enum
  ```rust
  /// Which hidden state to use for KV cache stash write.
  #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
  pub enum CacheStrategy {
      /// Use post-loop hidden state. Better for short structured generation.
      #[default]
      Last,
      /// Use pre-loop hidden state. Better for long CoT.
      First,
  }
  ```
- [x] T3: Add `TrainingFreeLoopConfig` struct
  ```rust
  /// Configuration for training-free loop wrapper.
  pub struct TrainingFreeLoopConfig {
      /// Loop window start layer index (inclusive).
      pub window_start: usize,
      /// Loop window end layer index (inclusive).
      pub window_end: usize,
      /// Number of sub-steps K (paper default: 2-3).
      pub loop_count: usize,
      /// Sub-step strategy (paper default: KStageRK β=0.5).
      pub strategy: SubStepStrategy,
      /// Iteration mode: block (dense) or layer (MoE).
      pub iteration_mode: IterationMode,
      /// KV cache strategy: first or last.
      pub cache_strategy: CacheStrategy,
  }
  ```
- [x] T4: Add `tf_loop` feature gate to `katgpt-core/Cargo.toml`

### Phase 1: Looped Forward with Sub-Stepping (katgpt-rs)
- [x] T5: Add `tf_loop` feature gate to `katgpt-rs/Cargo.toml` (depends on `lt2_looped`)
- [x] T6: Implement `forward_training_free_loop()` in `transformer.rs`
  - Pre-loop layers: standard forward, write KV normally
  - Loop body: K iterations with damped sub-stepping
  - Stash: single pass writes canonical KV
  - Post-loop layers: standard forward
- [x] T7: Implement block-mode sub-stepping (Algorithm 3 from paper)
  ```rust
  // Block-mode RK with anchor β
  let x_anchor = forward_block(x, window);  // one-shot for anchor
  let mut x = x_input;
  for k in 0..K {
      let y = forward_block(x, window);
      x = x + (1.0/K as f32) * (y - x);  // damped Euler
  }
  x = beta * x_anchor + (1.0 - beta) * x;  // anchor blend
  ```
- [x] T8: Implement layer-mode sub-stepping (per-layer variant of Algorithm 3)
  ```rust
  // Layer-mode: iterate each layer K times
  for layer in window_start..=window_end {
      let x_anchor = forward_layer(x, layer);
      for k in 0..K {
          let y = forward_layer(x, layer);
          x = x + (1.0/K as f32) * (y - x);
      }
      x = beta * x_anchor + (1.0 - beta) * x;
  }
  ```
- [x] T9: Implement `LoopMode::TrainingFree` dispatch variant
  - Extends existing `LoopMode` enum with new variant
  - Falls through to `forward_training_free_loop()` when selected

### Phase 2: KV Cache Snapshot/Restore
- [x] T10: Implement `snapshot_cache_lengths()` — record per-layer KV lengths
- [x] T11: Implement `restore_cache_lengths()` — crop KV back to snapshot
- [x] T12: Implement stash pass — single forward through window layers writes canonical KV
- [x] T13: Wire snapshot/restore into decode loop body
  - Prefill: loop body runs with `use_cache=false`, stash writes KV once
  - Decode: loop body runs with snapshot/restore per iteration, stash writes KV once

### Phase 3: Depth-Fraction Heuristic
- [x] T14: Implement `default_loop_window(n_layers: usize) -> (usize, usize)`
  ```rust
  pub fn default_loop_window(n_layers: usize) -> (usize, usize) {
      let center = (n_layers as f32 * 0.48) as usize;
      let start = center.saturating_sub(1);
      let end = (center + 2).min(n_layers - 1);
      (start, end)
  }
  ```
- [x] T15: Add TOML config parsing for training-free loop settings
  ```toml
  [model.tf_loop]
  enabled = true
  window_start = 12    # or auto from depth-fraction rule
  window_end = 15
  loop_count = 2       # K
  strategy = "KStageRK"
  beta = 0.5
  iteration_mode = "Block"  # or "Layer" for MoE
  cache_strategy = "First"
  ```

### Phase 4: GOAT Proof & Benchmarks
- [x] T16: GOAT proof: training-free loop produces finite, non-NaN logits — `proof_tf_loop_finite`
- [x] T17: GOAT proof: KV cache size identical to baseline (no growth with K) — `proof_tf_loop_cache_size`
- [x] T18: GOAT proof: bypass mode (prefill-only) throughput within ±5% of baseline — `proof_tf_loop_bypass_free`
- [x] T19: GOAT proof: layer-mode logits stable for K=2,3 — `proof_tf_loop_layer_mode_stable`
- [x] T20: Benchmark: training-free loop (K=2,3) vs baseline tok/s
- [x] T21: Benchmark: block-mode vs layer-mode on dense config
- [x] T22: Write benchmark results to `.benchmarks/034_tf_loop_goat.md`

### Phase 5: Documentation & Cleanup
- [x] T23: Update `README.md` with Training-Free Loop section
- [x] T24: Update `.docs/02_architecture.md` with sub-stepping forward pass diagram
- [x] T25: Run `cargo clippy --fix --allow-dirty` on all changed files

---

## Architecture

### Training-Free Loop Forward (Block-Mode)

```
Input: x ∈ R^{L×d}, window [a,b], K, β
Pre-loop: x ← L₀ ∘ ... ∘ L_{a-1}(x)     [standard, write KV]
Anchor:   x̃ ← (L_b ∘ ... ∘ L_a)(x)       [one-shot for β blend]
Loop:
  for k = 1..K:
    y ← (L_b ∘ ... ∘ L_a)(x)             [forward window]
    x ← x + (1/K)·(y - x)                [damped Euler sub-step]
  x ← β·x̃ + (1-β)·x                      [anchor blend]
Stash:    write canonical KV from x (cache=last) or x_pre (cache=first)
Post-loop: x ← L_{b+1} ∘ ... ∘ L_{N-1}(x) [standard, write KV]
Output: lm_head(x)
```

### Training-Free Loop Forward (Layer-Mode, for MoE)

```
Input: x ∈ R^{L×d}, window [a,b], K, β
Pre-loop: x ← L₀ ∘ ... ∘ L_{a-1}(x)
For each layer ℓ = a..b:
  x̃_ℓ ← L_ℓ(x)                           [per-layer anchor]
  for k = 1..K:
    y ← L_ℓ(x)                             [single layer forward]
    x ← x + (1/K)·(y - x)                 [damped Euler sub-step]
  x ← β·x̃_ℓ + (1-β)·x                    [per-layer anchor blend]
Stash: write canonical KV
Post-loop: x ← L_{b+1} ∘ ... ∘ L_{N-1}(x)
Output: lm_head(x)
```

### Decode-Time KV Cache Protocol

```
For each decode step:
  1. Snapshot: ℓ_i = |KV_cache[i]| for i in [a,b]
  2. Loop body (K iterations):
     a. Forward window with use_cache=true (reads past KV)
     b. Crop cache back to ℓ_i (zero net KV writes)
     c. Apply sub-step update to hidden state
  3. Stash: one forward pass writes canonical KV entry
  4. Post-loop: standard forward with KV writes
```

---

## Config

```toml
[model.tf_loop]
enabled = true
# Window: auto-computed from depth-fraction rule if not specified
# window_start = 12  # optional override
# window_end = 15    # optional override
loop_count = 2        # K (paper: 2 for dense, 3 for MoE)
strategy = "KStageRK" # only robust option
beta = 0.5            # anchor weight (0=full damped Euler, 1=identity)
iteration_mode = "Block"  # "Block" for dense, "Layer" for MoE
cache_strategy = "First"  # "First" for CoT, "Last" for short gen
decode_mode = "bypass"    # "bypass" (free), "full" (+22%), "first_n"
```

---

## Feature Gates

### katgpt-core/Cargo.toml
```toml
[features]
tf_loop = ["lt2_looped"]  # TrainingFreeLoopConfig, SubStepStrategy, IterationMode, CacheStrategy
```

### katgpt-rs/Cargo.toml
```toml
[features]
tf_loop = ["katgpt-core/tf_loop", "lt2_looped"]
```

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| **Plan 108 (LT2)** | Complementary. LT2 is training-time weight-sharing; TF-Loop is training-free sub-stepping. Both extend `LoopMode`. |
| **Plan 106 (DashAttention)** | TF-Loop layer-mode is needed when DashAttention is combined with MoE. |
| **Plan 105 (GDN2)** | GDN2's gating provides natural damping — TF-Loop may be less necessary on GDN2 layers. |
| **Plan 131 (SpecHop)** | SpecHop does multi-hop speculation; TF-Loop does single-token multi-pass refinement. Orthogonal. |

---

## Benchmark Plan

| Benchmark | Config | Metric | Expected |
|-----------|--------|--------|----------|
| `bench_tf_loop_bypass` | micro, TF-Loop K=2, bypass | tok/s | ±5% of baseline |
| `bench_tf_loop_full` | micro, TF-Loop K=2, full decode | tok/s | ~80% of baseline |
| `bench_tf_loop_layer_mode` | micro, TF-Loop K=2, layer-mode | tok/s | ~90% of block-mode |
| `bench_tf_loop_cache` | micro, TF-Loop K=3 | KV entries | Identical to baseline |
| `bench_tf_loop_cosine` | micro, TF-Loop K=2 vs baseline | cos-sim | >0.95 (sub-step is refinement) |

### GOAT Proof Criteria

1. **Stability**: All logits finite, non-NaN at K=2, K=3, K=4
2. **Cache size**: KV cache identical to baseline (no growth with K)
3. **Bypass throughput**: Within ±5% of baseline
4. **No regression**: Non-`tf_loop` builds unchanged
5. **Layer-mode stability**: Layer-mode K=2,3 produces finite logits

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| No quality gain on our micro model | Medium | Paper shows gains on ≥1.7B; micro may be too small. Still useful for MoE/routing integration. |
| Snapshot/restore overhead | Low | Zero-allocation, just length tracking. Negligible. |
| Block-mode instability on MoE | High | Default to layer-mode when MoE detected. |
| β=0.5 not optimal for our model | Low | Paper shows broad robustness to β in [0,1]. Configurable. |

---

## Implementation Order

```
Phase 0: Core types (katgpt-core)    [~2h]
Phase 1: Looped forward + sub-step   [~4h]  ← main work
Phase 2: KV cache snapshot/restore   [~2h]
Phase 3: Depth-fraction heuristic    [~1h]
Phase 4: GOAT proof & benchmarks     [~2h]
Phase 5: Docs & cleanup              [~1h]
────────────────────────────────────────────
Total estimate:                      ~12h
```

---

## References

- Paper: https://arxiv.org/abs/2605.23872
- Our LT2 (Research 073, Plan 108): training-time looped transformers
- Deep Equilibrium Models (Bai et al., 2019): ODE interpretation
- Algorithm 3 (K-stage RK): the only robust sub-stepping strategy
