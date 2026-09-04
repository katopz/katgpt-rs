# Benchmark 456 — Interpolation Geometry GOAT Gate (Issue 158 Phase 1)

> **Primitive:** `katgpt_core::interpolation_geometry`
> **Feature:** `interpolation_geometry` (opt-in)
> **Bench:** `crates/katgpt-core/benches/bench_456_interpolation_geometry_goat.rs`
> **Source:** `Issue 158`,
> [Research 445](../.research/445_Latent_Thought_Flows_Text_Compression.md) —
> Prabhudesai & Geng, *Latent Thought Flows with Text Compression* (Jun 2, 2026).

## TL;DR

**All GOAT gates PASS.** The `interpolation_geometry` primitive (generic
`LatentSpace` trait + `imauve_score` + `intervention_battery`) meets the
audit-cadence contract on the synthetic shape analogs. **NOT promoted to
default-on** — this is an evaluation methodology, not a runtime primitive;
promotion requires a real-substrate audit (riir-engine `NpcEmotionScalars`,
riir-neuron-db `NeuronShard`) that either confirms all substrates have good
geometry or surfaces a fix-worthy failure.

## Gates

| Gate | Target | Measured (release, Apple Silicon) | Verdict |
|---|---|---|---|
| **G1 correctness** | `good_score > bad_score`, `good > 0.9`, `bad < 0.95` | good=**0.9646**, bad=**0.8087** | **PASS** |
| **G2 perf** | `imauve_score` at n=256 × d=64 < 50 ms | **642 µs** median (78× headroom) | **PASS** |
| **G3 no-regression** | `cargo clippy --all-features` clean | clean | **PASS** |
| **G4 zero-alloc** | 0 allocs / 100 calls × 2 primitives | **0 / 0** | **PASS** |

## G1 details — the headline correctness check

The test constructs two synthetic 2D Gaussian-mixture spaces with known
interpolation geometries:

- **Good geometry** (`good_along_manifold`): 8 cluster centers along the
  1D manifold `y = x`. Midpoints of cross-cluster neighbors stay on the
  manifold.
- **Bad geometry** (`bad_radial_clustering`): 8 cluster centers on a circle
  of radius 5, evenly spaced in angle. Midpoints of arc-adjacent neighbors
  fall INSIDE the circle (the paper's "length clustering" failure mode
  analog).

The iMAUVE score must distinguish them:

```
good (1D manifold):   score = 0.964645  (n_anchors = 8)
bad  (radial cluster): score = 0.808658  (n_anchors = 8)

good > bad:    PASS
good > 0.9:    PASS  (got 0.9646447)
bad  < 0.95:   PASS  (got 0.8086583)
```

**Key design decision:** use 1 point per cluster so the nearest neighbor is
forced to be cross-cluster. With multiple points per cluster + small noise,
the nearest neighbor is always within-cluster and the geometry difference is
masked. This mirrors the paper's protocol — for each REAL example, find its
nearest neighbor in the latent space (which is typically a DIFFERENT example,
not the same one).

## G2 details — audit-cadence latency

The reference scale is the paper's TinyStories n≈256, applied at shard
`style_weights` dimensionality (d=64):

```
Config: n=256 anchors, dim=64 (NeuronShard::style_weights scale)
       21 timed runs (median), 10 warmup, seed=42

median over 21 runs: 642.75 µs  (target: < 50.00 ms)
```

The 50 ms target is an audit-cadence budget (vs Plan 342's 5 µs hot-path
budget). 642 µs gives **78× headroom** — sufficient for a future router
integration that runs the metric at every-K-ticks cadence.

The point cloud: 8 clusters along a 1D manifold embedded in 64D, with 32
points per cluster at small Gaussian noise (σ=0.01). All points have all 64
coordinates equal (modulo noise) — the manifold is the diagonal of the
64-cube. Midpoints stay on this diagonal → score > 0.95.

## G4 details — zero-alloc hot path

The protocol is designed to be alloc-free on the hot path:

- `imauve_score` takes a caller-supplied `midpoint_scratch: &mut Point`,
  reused across all anchors. Zero per-anchor allocation.
- `intervention_battery` takes three caller-supplied scratch buffers
  (`zero_scratch`, `mean_scratch`, `noise_scratch`), reused across the five
  interventions. Zero per-intervention allocation.

Measured via the shared `CountingAllocator`:

```
imauve_score × 100 calls:           0 allocs, 0 deallocs  (target: 0, 0)
intervention_battery × 100 calls:   0 allocs, 0 deallocs  (target: 0, 0)
```

## Why NOT promoted to default-on

Per the GOAT promotion rule (AGENTS.md "Feature Flag Discipline"), promotion
requires a modelless **gain** — a measurable improvement in latency,
quality, or security. This primitive is an **evaluation methodology**: it
produces *measurements*, not gains. The metric is the regression test for a
FUTURE fix (if any substrate has bad interpolation geometry), not a runtime
improvement itself.

The promotion path:

1. riir-engine implements `impl LatentSpace for NpcEmotionScalars` (wraps
   the existing decode-to-5-scalars bridge).
2. riir-neuron-db implements `impl LatentSpace for NeuronShard` (wraps the
   style_weights decode).
3. Run the iMAUVE score on real substrates:
   - If **all pass** → close Issue 158, document the methodology, no fix
     needed.
   - If **any fails** → open a plan for a modelless fix (regularization,
     projection, or commitment cadence change). The fix becomes the GOAT
     candidate; the metric stays as the regression test.

Until that audit runs, the primitive stays opt-in (feature
`interpolation_geometry`). The trait + reference impls are stable; the
feature gate exists so consumers don't pay for an evaluation methodology
they're not using.

## How to reproduce

```bash
# Run the GOAT bench (release mode required for G2 perf).
cargo bench -p katgpt-core --features interpolation_geometry --no-default-features \
    --bench bench_456_interpolation_geometry_goat

# Run the unit tests.
cargo test -p katgpt-core --features interpolation_geometry --no-default-features \
    interpolation_geometry::

# Clippy (G3 no-regression).
cargo clippy -p katgpt-core --features interpolation_geometry --no-default-features --lib
cargo clippy -p katgpt-core --all-features --lib
```

## See also

- `Issue 158` —
  the PoC issue with 4-phase breakdown + three-pressure audit.
- [Research 445](../.research/445_Latent_Thought_Flows_Text_Compression.md) —
  the parent research note with the full paper distillation.
- [.docs/04_calibration/interpolation_geometry.md](../.docs/04_calibration/interpolation_geometry.md) —
  the user-facing API doc.
- [Bench 342 (latent_trajectory_geometry)](342_latent_trajectory_geometry_gate.md) —
  sibling diagnostic (probe-free trajectory geometry vs interpolation geometry).
