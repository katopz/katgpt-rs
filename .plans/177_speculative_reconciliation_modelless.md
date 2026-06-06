# Plan 177: Speculative Reconciliation Engine — Modelless Implementation

> **Status:** ✅ Complete — GOAT G1-G8 — Default ON
> **Research:** [Research 156](../.research/156_Speculative_Reconciliation_Engine.md)
> **Depends On:** Plan 032 ✅ (HL), Plan 053 ✅ (DeltaMemory), Plan 155 ✅ (LEO), Plan 194 ✅ (Adaptive CoT)
> **Feature Gate:** `spec_reconciliation`
> **Target:** `katgpt-rs/src/spec_reconciliation/`

---

## Overview

Implement the modelless Speculative Reconciliation Engine. This is the core inference primitive that verifies offline game state trajectories against a LEO-generated plausibility manifold. Lives entirely in `katgpt-rs` (MIT engine).

---

## Task

- [x] **T1: Core Types** — `TrajectoryPoint` (fixed-layout `[f32; 8]`), `ReconciliationVerdict` enum (Accept/Quarantine/Uncertain), `ReconciliationConfig` (thresholds, K, max_speed, map_bounds). All `#[repr(C)]`, zero heap.
  - File: `katgpt-rs/src/spec_reconciliation/types.rs`
  - Test: unit test for TrajectoryPoint construction, Verdict Debug/Clone

- [x] **T2: `ReconciliationPruner`** — Implements `ConstraintPruner`. Hard bounds: velocity (max_speed × dt), position (map_bounds), kill_rate (Chebyshev 5σ). Returns `bool`.
  - File: `katgpt-rs/src/spec_reconciliation/reconciliation_pruner.rs`
  - Reuses: `ConstraintPruner` trait from `katgpt-core`
  - Test: G1 (velocity invariant), G2 (position invariant), G3 (kill-rate bound)

- [x] **T3: `ManifoldGenerator` trait** — `fn generate(h_last, q_goals, K, dt, rng) -> [TrajectoryPoint; K]`. Default impl: LEO-weighted goal sampling + physics + Gaussian noise. No neural forward pass.
  - File: `katgpt-rs/src/spec_reconciliation/manifold.rs`
  - Reuses: `LeoHead` trait for Q-values, HLA hidden state format
  - Test: verify K=16 trajectories generated, verify σ grows with dt, verify goals sampled from LEO

- [x] **T4: `ManifoldScorer`** — Implements `ScreeningPruner`. Computes `max_j(cosine_sim(T_client, T_spec[j]))`. SIMD-optimized cosine similarity.
  - File: `katgpt-rs/src/spec_reconciliation/manifold_scorer.rs`
  - Reuses: `ScreeningPruner` trait from `katgpt-core`, existing SIMD cosine similarity
  - Test: known vector pairs → expected similarity scores

- [x] **T5: `SpecReconciler`** — Orchestrates T2-T4. Implements `SpeculativeVerifier` adapted for trajectories. Pipeline: hard bounds → manifold generation → soft scoring → verdict.
  - File: `katgpt-rs/src/spec_reconciliation/reconciler.rs`
  - Reuses: `SpeculativeVerifier` trait pattern
  - Test: end-to-end with legitimate trajectory → Accept, hack trajectory → Quarantine

- [x] **T6: `AdaptiveReconciler`** — Wraps `SpecReconciler` with `BanditPruner<ManifoldScorer>` for per-player threshold learning. Uses `ThinkingBandit` freeze/thaw for persistence across sessions.
  - File: `katgpt-rs/src/spec_reconciliation/adaptive.rs`
  - Reuses: `BanditPruner` from `katgpt-rs/src/pruners/bandit.rs`, `ThinkingBanditFrozen` from Plan 194
  - Test: simulate 100 reconciliations, verify bandit converges to optimal threshold

- [x] **T7: GOAT Proof Suite** — 8 formal verification gates (G1-G8 from Research 156 §5.1).
  - File: `katgpt-rs/tests/spec_reconciliation_proof.rs`
  - G1: velocity invariant (property test)
  - G2: position invariant (property test)
  - G3: kill-rate bound (property test)
  - G4: manifold coverage (Monte Carlo, 10K trajectories, >95% within manifold)
  - G5: latency bound (micro-benchmark, <1ms for <5min offline)
  - G6: false positive rate (<1% legitimate quarantined)
  - G7: false negative rate (>99% hacks quarantined)
  - G8: matrix soundness (determinant audit on all accepted merges)

- [x] **T8: Feature Gate + Module Wiring** — `spec_reconciliation` feature gate in `katgpt-rs/Cargo.toml`. Module public API. Zero impact when disabled.
  - File: `katgpt-rs/src/spec_reconciliation/mod.rs`, `katgpt-rs/Cargo.toml`
  - Test: `cargo build` without feature → no compilation. `cargo build --features spec_reconciliation` → compiles.

- [x] **T9: Example Demo** — Interactive demo showing reconciliation of 4 scenarios: legitimate play, teleport hack, kill-rate hack, direction mismatch.
  - File: `katgpt-rs/examples/spec_reconciliation_demo.rs`
  - Test: `cargo run --example spec_reconciliation_demo --features spec_reconciliation`

- [x] **T10: Benchmark** — Micro-benchmark: reconciliation latency vs offline duration (1s, 10s, 60s, 300s, 600s). Report P50/P99.
  - File: `katgpt-rs/tests/spec_reconciliation_bench.rs`
  - Test: all durations <1ms P50

---

## Dependencies

```
T1 → T2, T3, T4
T2 → T5
T3 → T5
T4 → T5, T6
T5 → T6, T7, T9, T10
T6 → T7
T7 → T8
T8 → T9, T10
```

Parallelizable: T2 + T3 + T4 can run concurrently after T1.

---

## Default Behavior

Per constraint "if gain and no perf hurt must be on by default":
- Promoted to default-on after GOAT proof (G5 latency <1ms, G6 false positive <1%). All tests pass.
- Performance audit: verify no regression on existing speculative decoding benchmarks when feature enabled
