# Plan 242: Fourier-Smoothed Potential Fields for LEO Crowd Flow

**Status:** GOAT-Gated (Marginal — Needs Benchmark Proof)
**Feature Flag:** `flow_field_nav` (opt-in, requires `leo_all_goals`)
**Routing:** katgpt-rs → `crates/katgpt-core/src/flow/`
**Research Origin:** Plan 212 Pillar 7 (Marginal Gemini)
**Depends On:** Plan 155 (LeoHead, `leo_all_goals`), Plan 156 (spectral_hierarchy FFT)

## Why

When 100+ NPCs share the same goal (e.g., "go to town square"), running individual LEO Q-value lookups per NPC per tick is wasteful — same gradient, same obstacles, same destination. A shared 2D flow field computed once per tick, FFT-smoothed to eliminate local minima, lets all NPCs read their gradient direction via O(1) lookup instead of per-entity pathfinding. This is the classic continuum crowds idea (Treuille et al. 2006) adapted to our LEO Q-value framework.

**Key insight:** LEO already computes Q-values per goal. `LeoHead::all_goals_q()` produces `[goals × actions]`. For spatial goals, the max-Q action per cell IS a flow vector. FFT smoothing the resulting potential field removes discretization noise and local minima — exactly what `spectral_hierarchy`'s Jacobi/Haar pipeline already does conceptually.

**Why marginal:** Only helps for crowd scenarios (many entities, shared goals). Individual explorers or small groups (<20) won't benefit. The FFT compute cost must amortize over enough NPCs. May not be worth the module complexity.

## Architecture

```mermaid
graph TD
    LEO[LeoHead::all_goals_q] --> QGrid[LeoPotentialGrid]
    QGrid --> |per goal| Raw[Raw Q-value grid]
    Raw --> FFT[FFT Smooth]
    FFT --> |low-pass filter| Gradient[Gradient Field]
    Gradient --> FlowField[FlowField dx,dy per cell]

    Obs[Dynamic Obstacle] --> |mark blocked cells| Raw
    Obs --> |trigger recompute| FFT

    NPC[NPC at cell x,y] --> |O(1) lookup| FlowField
    FlowField --> |dx,dy| Move[Steering Vector]
    Move --> |blend with avoidance| FinalMove[Final Movement]

    subgraph GOAT
        Bench[Benchmark: 100 NPCs shared goal]
        Bench --> |CPU time < 80% of individual| Pass[Promote to default]
        Bench --> |CPU time >= 80%| Kill[Kill Plan]
    end
```

### Data Structures

```rust
/// 2D grid of flow vectors — preferred movement direction per cell.
/// One per goal (or per goal-group for shared goals).
#[repr(C)]
pub struct FlowField {
    /// Width in cells.
    pub w: u16,
    /// Height in cells.
    pub h: u16,
    /// Flow vectors: [w * h * 2] — (dx, dy) per cell, row-major.
    /// Normalized to unit length or zero for blocked cells.
    flow: Vec<f32>,
}

/// Maps LEO Q-values onto a 2D spatial grid for a specific goal.
/// Intermediate structure — consumed by FFT smoothing.
pub struct LeoPotentialGrid {
    pub w: u16,
    pub h: u16,
    /// Q-values: [w * h] — max-Q or expected value per cell for one goal.
    potential: Vec<f32>,
    /// Blocked cells (obstacles, walls). Bitfield, 1 = blocked.
    blocked: Vec<u64>,
}

/// FFT smoothing parameters.
pub struct FlowFieldConfig {
    /// Low-pass cutoff frequency (fraction of Nyquist). Default: 0.25.
    pub cutoff: f32,
    /// Obstacle inflation radius (cells). Default: 1.
    pub obstacle_radius: u8,
    /// Minimum gradient magnitude to produce a flow vector. Default: 1e-4.
    pub min_gradient: f32,
    /// Recompute threshold: how many cells must change to trigger FFT. Default: 5.
    pub dirty_threshold: u16,
}
```

### Pipeline

1. **Q-value extraction:** For each goal with ≥`min_npcs` assigned NPCs, call `LeoHead::all_goals_q()` for cells near the goal region. Build `LeoPotentialGrid` — max-Q per cell.
2. **FFT smooth:** Forward FFT → low-pass filter (zero out high frequencies above cutoff) → inverse FFT. This removes local minima and creates smooth gradients toward goals.
3. **Gradient computation:** Finite differences on smoothed potential → (dx, dy) per cell. Normalize to unit vectors.
4. **Obstacle handling:** Blocked cells get zero flow. Obstacle inflation via morphological dilation before FFT to prevent flow into walls.
5. **NPC lookup:** Each NPC reads `flow[y * w + x]` — O(1). Blend with local avoidance (separation, obstacle proximity).

### Integration Points

- **`LeoHead` trait (Plan 155):** Source of Q-values. `q_for_goal()` extracts per-goal slice.
- **`spectral_hierarchy` (Plan 156):** FFT primitives already exist (Jacobi eigendecomposition uses spectral methods). Reuse `rustfft` or add minimal DFT if not already present.
- **`FlowField` storage:** One per active goal with ≥N NPCs. Invalidated when obstacles change.
- **NPC tick loop:** Replace individual LEO Q-lookup with flow field lookup when `flow_field_nav` enabled.

## Tasks

### T1: Core Types + FlowField
- [x] Create `crates/katgpt-core/src/flow/mod.rs` with `FlowField`, `LeoPotentialGrid`, `FlowFieldConfig`
- [x] `FlowField::lookup(x, y) -> (f32, f32)` — O(1) indexed access with bounds check
- [x] `FlowField::is_blocked(x, y) -> bool` — obstacle query
- [x] Unit tests: lookup, bounds, blocked cells

### T2: LeoPotentialGrid Builder
- [x] `LeoPotentialGrid::from_q_values()` — map LEO Q-values onto 2D grid
- [x] `LeoPotentialGrid::mark_blocked()` — set obstacle cells from spatial data
- [x] `LeoPotentialGrid::gradient() -> FlowField` — finite differences + normalization
- [x] Unit tests: grid construction, gradient direction correctness

### T3: FFT Smoothing
- [x] `fft_smooth(grid: &mut [f32], w: usize, h: usize, cutoff: f32)` — forward FFT, low-pass, inverse
- [x] Use `rustfft` crate (add to deps, feature-gated under `flow_field_nav`)
- [x] Obstacle inflation before FFT (morphological dilation)
- [x] Unit tests: smooth field has no local minima, gradient points toward goal

### T4: FlowFieldCache — Per-Goal Shared Cache
- [x] `FlowFieldCache` — maps `(goal_id) -> FlowField`, recomputes when dirty
- [x] Dirty tracking: count changed cells since last compute. Recompute when ≥ threshold.
- [x] Integration with `LeoHead`: extract Q-values on recompute
- [x] Unit tests: cache hit/miss, dirty threshold, recompute trigger

### T5: NPC Integration
- [x] `flow_steering(field: &FlowField, pos: (f32, f32)) -> (f32, f32)` — bilinear interpolation for sub-cell positions
- [x] Blend flow vector with separation/avoidance forces
- [x] Fallback to individual LEO when NPC is off-grid or goal has < min_npcs
- [x] Integration test: 10 NPCs with shared goal reach target via flow field

### T6: Feature Gate
- [x] Add `flow_field_nav` feature to `crates/katgpt-core/Cargo.toml` (requires `leo_all_goals`)
- [x] Add `flow_field_nav` feature to workspace `Cargo.toml` (opt-in, NOT default)
- [x] Gate all `flow/` module behind `#[cfg(feature = "flow_field_nav")]`
- [x] Conditionally compile `rustfft` dependency

### T7: GOAT Benchmark
- [x] Create `crates/katgpt-core/benches/flow_field_bench.rs`
- [x] Benchmark A: 100 NPCs, 1 shared goal, individual LEO Q-lookup per tick (baseline)
- [x] Benchmark B: 100 NPCs, 1 shared goal, shared FlowField lookup per tick
- [x] Metric 1: Total CPU time for 100 ticks (lower is better)
- [x] Metric 2: FFT smoothing cost by grid size (32, 64, 128)
- [x] Metric 3: Dynamic obstacle response — time to re-converge after obstacle insertion
- [x] Criterion: >20% CPU improvement required to promote

### T8: Dynamic Obstacle Response
- [x] On obstacle change event, mark affected cells dirty in `FlowFieldCache`
- [x] Trigger FFT recompute when dirty threshold exceeded (lazy via `get_or_compute`)
- [x] NPCs seamlessly pick up new flow direction on next lookup
- [x] Test: insert wall, verify NPCs reroute (covered by `bench_dynamic_obstacle` + `test_dirty_tracking_invalidates`)

## GOAT Gate

**Target:** >20% CPU improvement over individual LEO pathfinding for ≥100 entities with shared goals.

| Metric | Baseline (individual LEO) | Flow Field | Required |
|--------|---------------------------|------------|----------|
| CPU time (100 NPCs, 100 ticks) | X μs | Y μs | Y < 0.8×X |
| Path quality (avg steps) | N steps | ≤ 1.1×N | No degradation |
| Dynamic obstacle reconvergence | — | ≤ 2 ticks | Must converge |
| Break-even entity count | — | ≤ 100 NPCs | Threshold for win |

**Promotion:** If GOAT passes → add `flow_field_nav` to default features, update Cargo.toml comment with GOAT result.
**Demotion:** If GOAT fails → keep feature opt-in, document why.

## Kill Condition

Close this plan if ANY of:
- CPU improvement <20% for 100 NPCs (not worth the complexity)
- Break-even only at >500 NPCs (too niche — most game scenes have <200 NPCs per goal)
- Path quality degrades >10% (smoothed gradients lead NPCs into suboptimal paths)
- FFT recompute latency >2ms for 128×128 grid (defeats real-time purpose)
- Module adds >500 lines without proportional gain

**Verdict:** This is a marginal optimization. If the numbers don't clearly justify the additional module, kill it. Individual LEO with `leo_all_goals` already works well for small groups.

## Expected Result

If GOAT passes:
- `flow_field_nav` feature in default set
- 100+ NPCs with shared goals navigate via O(1) flow lookup
- Dynamic obstacles handled via FFT recompute (≤2 tick delay)
- Clean integration with existing `LeoHead` trait — no trait changes needed
- ~400-600 lines in `crates/katgpt-core/src/flow/`

If GOAT fails:
- Feature remains opt-in (or removed)
- Document benchmark results in `.issues/` for future reference
- Individual LEO pathfinding remains the default

## TL;DR

Fourier-smoothed flow fields for crowd NPCs sharing goals. LEO Q-values → 2D grid → FFT smooth → gradient → O(1) NPC lookup. GOAT-gated: needs >20% CPU win for 100 NPCs or we kill it. Feature `flow_field_nav`, requires `leo_all_goals`.
