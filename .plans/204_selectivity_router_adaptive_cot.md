# Plan 204: Self-Learning Selectivity Router — Adaptive CoT

**Date**: 2026-06-07
**Status**: 📋 Plan
**Research**: `.research/180_Rosetta_Scaling_Polarization_Data_Filtering.md` (Section 3.1)
**Extracted From**: Plan 203 (Phase 2.2 — Selectivity Router)
**GOAT Rank**: #1 (self-learning, zero training, adaptive CoT)
**Feature Gate**: `selectivity_router`

---

## Background

Research 180 proves that **selectivity (excess kurtosis) increases predictably with scale and training**. As models learn, individual neuron/logit marginals transition from flat/polysemantic distributions to peaked/monosemantic ones.

This gives us a **dynamic, zero-cost signal** for routing between "thinking" (Chain-of-Thought) and "non-thinking" (direct) inference modes — without any LLM training.

### The Key Insight

- Positions that become **more selective** (high kurtosis) → model is confident → **direct mode**
- Positions that remain **polysemantic** (low kurtosis) → model needs exploration → **CoT mode**
- **Self-improving**: as the model learns (or as we serve more requests), routing improves automatically
- Maps to constraint: **self-learning adaptive CoT without LLM training**

---

## Architecture

### Core Struct

```rust
/// Per-position selectivity router using the polarization effect.
///
/// High kurtosis (selective/monosemantic) → direct mode (no thinking).
/// Low kurtosis (polysemantic) → CoT mode (thinking needed).
///
/// Self-learning: observes kurtosis at each position across inference
/// requests. As the model (or domain) changes, the routing adapts.
#[cfg(feature = "selectivity_router")]
pub struct SelectivityRouter {
    /// Per-position EMA of excess kurtosis.
    /// Grows dynamically, pre-allocate with `with_capacity()`.
    position_kurtosis: Vec<f32>,
    /// Threshold for direct vs CoT routing.
    /// kurtosis ≥ threshold → direct mode.
    /// kurtosis < threshold → CoT mode.
    kurtosis_threshold: f32, // default: 1.0
    /// EMA decay factor. Lower = slower adaptation.
    alpha: f32, // default: 0.1
}
```

### API Surface

```rust
#[cfg(feature = "selectivity_router")]
impl SelectivityRouter {
    /// Create a new router with default thresholds.
    pub fn new() -> Self;

    /// Create with pre-allocated capacity for `max_positions` positions.
    pub fn with_capacity(max_positions: usize) -> Self;

    /// Should this position use CoT (thinking) mode?
    /// Returns `true` if kurtosis is LOW → polysemantic → needs thinking.
    /// Returns `false` if kurtosis is HIGH → monosemantic → direct answer.
    ///
    /// O(1) — single array lookup + comparison.
    pub fn should_think(&self, position: usize) -> bool;

    /// Observe kurtosis at a given position. Updates EMA.
    /// Call after each speculative decode step with the computed kurtosis.
    ///
    /// O(1) amortized — Vec resize only when new positions encountered.
    pub fn observe(&mut self, position: usize, kurtosis: f32);

    /// Get the current EMA kurtosis for a position.
    /// Returns `None` if position has never been observed.
    pub fn kurtosis_at(&self, position: usize) -> Option<f32>;

    /// Reset all tracking state. Use when switching domains or sessions.
    pub fn reset(&mut self);

    /// Save kurtosis profile to bytes (for persistence).
    pub fn serialize(&self) -> Vec<u8>;

    /// Load kurtosis profile from bytes (for cold start recovery).
    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError>;
}
```

### Internal Logic

```rust
impl SelectivityRouter {
    pub fn should_think(&self, position: usize) -> bool {
        let k = self.position_kurtosis.get(position).copied().unwrap_or(f32::MAX);
        // No data yet → treat as high kurtosis (direct mode, optimistic)
        // Low kurtosis → polysemantic → needs thinking
        k < self.kurtosis_threshold
    }

    pub fn observe(&mut self, position: usize, kurtosis: f32) {
        if position >= self.position_kurtosis.len() {
            self.position_kurtosis.resize(position + 1, 0.0);
        }
        let prev = self.position_kurtosis[position];
        self.position_kurtosis[position] = self.alpha * kurtosis + (1.0 - self.alpha) * prev;
    }
}
```

### CPU/GPU Auto-Route Integration

The router also feeds into CPU/GPU routing:
- **High selectivity positions** → CPU can handle (predictable, peaked distributions)
- **Low selectivity positions** → GPU needed (complex, flat distributions)
- This maps to constraint #6 (CPU/GPU auto-route when load changes)

```rust
/// Route recommendation based on position selectivity.
pub enum ComputeRoute {
    /// High kurtosis → predictable → CPU speculative
    CpuSpeculative,
    /// Low kurtosis → complex → GPU autoregressive
    GpuAutoregressive,
}

impl SelectivityRouter {
    /// Recommend compute route for a position.
    pub fn recommend_route(&self, position: usize) -> ComputeRoute {
        match self.should_think(position) {
            true => ComputeRoute::GpuAutoregressive,   // needs CoT → GPU
            false => ComputeRoute::CpuSpeculative,      // direct → CPU
        }
    }
}
```

---

## File Layout

```
crates/katgpt-core/src/
├── polarization/
│   ├── mod.rs                  — existing, add selectivity_router cfg
│   ├── selectivity_router.rs   — NEW: SelectivityRouter + ComputeRoute
│   ├── kurtosis.rs             — existing: excess_kurtosis() (Plan 203 Phase 1)
│   └── polarization_index.rs   — existing: PolarizationIndex (Plan 203 Phase 1)
```

---

## Tasks

### Implementation

- [ ] Create `crates/katgpt-core/src/polarization/selectivity_router.rs`
  - [ ] Implement `SelectivityRouter` struct with `position_kurtosis: Vec<f32>`, `kurtosis_threshold: f32`, `alpha: f32`
  - [ ] Implement `new()` with defaults (threshold=1.0, alpha=0.1)
  - [ ] Implement `with_capacity(max_positions: usize)` for pre-allocation
  - [ ] Implement `should_think(position) -> bool` — O(1) lookup, low kurtosis → CoT
  - [ ] Implement `observe(position, kurtosis)` — per-position EMA update
  - [ ] Implement `kurtosis_at(position) -> Option<f32>` — read current EMA
  - [ ] Implement `reset()` — clear all tracking
  - [ ] Implement `ComputeRoute` enum (`CpuSpeculative`, `GpuAutoregressive`)
  - [ ] Implement `recommend_route(position) -> ComputeRoute`
- [ ] Wire `selectivity_router` module into `polarization/mod.rs` behind `#[cfg(feature = "selectivity_router")]`

### Integration

- [ ] Add integration point: after each speculative decode, call `router.observe(position, excess_kurtosis(logits))` with computed kurtosis
- [ ] Add integration point: before generation, check `router.should_think(position)` → route direct vs CoT
- [ ] Wire `recommend_route()` into CPU/GPU dispatch (if applicable to current inference pipeline)

### Persistence

- [ ] Implement `serialize() -> Vec<u8>` — bincode or simple f32 slice dump
- [ ] Implement `deserialize(data: &[u8]) -> Result<Self, ProfileError>`
- [ ] Add `ProfileError` enum (`InvalidMagic`, `VersionMismatch`, `TruncatedData`)
- [ ] Add save/load to disk helper: `save_profile(path: &Path)` / `load_profile(path: &Path)`
- [ ] Cold start recovery: load saved profile on startup, falls back to fresh router if no file

### Feature Gate

- [ ] Add `selectivity_router = []` feature to `Cargo.toml`
- [ ] All new types behind `#[cfg(feature = "selectivity_router")]`
- [ ] Add to `polarization_all` bundle feature
- [ ] GOAT gate: default-on after verification (see below)

### Tests

- [ ] Test: fresh router — no observations → `should_think` returns `false` for all positions (optimistic direct mode, since kurtosis defaults to MAX)
- [ ] Test: after observing high kurtosis (3.0+) → `should_think` returns `false` (direct mode)
- [ ] Test: after observing low kurtosis (0.0-) → `should_think` returns `true` (CoT mode)
- [ ] Test: EMA convergence — recent observations dominate over old ones
- [ ] Test: router converges to correct routing after N observations (N=100)
- [ ] Test: cold start from saved profile — serialize → deserialize → identical routing decisions
- [ ] Test: `recommend_route()` maps correctly to `ComputeRoute` variants
- [ ] Test: `with_capacity()` pre-allocates without reallocation

### Benchmarks

- [ ] Benchmark: `should_think()` overhead < 100ns per decision (target: O(1) lookup)
- [ ] Benchmark: `observe()` overhead < 100ns per call (target: O(1) amortized)
- [ ] Benchmark: `serialize()` / `deserialize()` on profiles with 1K, 10K, 100K positions

### Example

- [ ] Add example: `examples/selectivity_router_demo.rs`
  - Simulate N inference requests with varying kurtosis patterns
  - Show before/after: thinking tokens used vs without router
  - Print routing convergence over time

---

## GOAT Verification

| Metric | Threshold | How to Measure |
|--------|-----------|----------------|
| CoT token reduction | ≥ 20% fewer thinking tokens on mixed-domain workload | Run inference with/without router, count CoT tokens |
| Routing decision latency | < 100ns per `should_think()` call | `cargo bench --features selectivity_router` |
| Convergence | Router stabilizes within 100 observations per position | Unit test with synthetic data |
| Cold start | Saved profile restores identical routing | Serialize → deserialize → assert_eq routing |
| No perf hurt | Inference throughput ≤ 1% slower with router enabled | Benchmark with/without feature flag |

### GOAT Status

- **GOAT**: Yes — self-learning, zero training cost, maps directly to adaptive CoT constraint
- **Default**: ON after GOAT proof (verify no perf hurt via benchmarks)
- **Feature gate**: `selectivity_router`

---

## Relationship to Plan 203

This plan extracts **Phase 2.2 (Selectivity Router)** from Plan 203 into a standalone plan for focused implementation. Plan 203 covers the full Rosetta Scaling polarization suite (6 components); this plan deep-dives on just the adaptive CoT router.

| Aspect | Plan 203 (Phase 2.2) | This Plan (204) |
|--------|----------------------|------------------|
| Scope | 1 task block | Full implementation detail |
| Persistence | Not mentioned | Save/load + cold start |
| CPU/GPU routing | Mentioned in PolarizationIndex | `ComputeRoute` enum + integration |
| Tests | Basic convergence test | 8+ unit tests + benchmarks |
| Example | Not mentioned | Demo example |

### Dependencies on Plan 203 Phase 1

This router depends on `excess_kurtosis()` from Plan 203 Phase 1.1 (`kurtosis_gate`). If that isn't implemented yet, the router can accept pre-computed kurtosis values — the kurtosis computation is decoupled from the routing decision.

---

## Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `kurtosis_threshold` | 1.0 | [0.5, 3.0] | Lower = more CoT (conservative), higher = more direct (aggressive) |
| `alpha` (EMA decay) | 0.1 | [0.01, 0.5] | Lower = slower adaptation (more stable), higher = faster tracking |

---

## TL;DR

**Self-learning adaptive CoT router from Research 180's polarization effect.** Tracks per-position EMA kurtosis across inference requests — high kurtosis (monosemantic/confident) routes to direct mode, low kurtosis (polysemantic/uncertain) routes to CoT mode. Self-improving as the model serves more requests. Includes persistence for cold start, CPU/GPU compute routing, and feature-gated behind `selectivity_router`. GOAT gate: ≥ 20% CoT token reduction, < 100ns per decision. Extracted from Plan 203 Phase 2.2 for focused implementation.
