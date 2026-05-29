# Plan 159: FlashLib Dual-Gram Routing + Roofline Cost Model

**Date:** 2026-05-28
**Status:** Complete
**Research:** R130 (FlashLib GPU Classical ML Operators)
**Feature Gates:** `dual_gram_pca` (default-ON), `roofline_cost` (default-ON)
**Promoted:** After GOAT proof — both default-ON
**Cross-ref:** riir-ai Plan 159 (game-specific calibration thresholds)

---

## Context

FlashLib (Yang et al., 2026) uses dual-Gram PCA routing and roofline cost prediction to accelerate classical ML operators on GPU. After source code audit (R130), two techniques transfer directly to our wgpu/Rust stack:

1. **Dual-Gram Routing** — When `seq_len < 4 * d_h`, compute X·Xᵀ (seq_len²) instead of Xᵀ·X (d_h²). Up to 512× speedup on SpectralQuant calibration for short sequences.
2. **Roofline Cost Model** — Port FlashLib's `info/roofline.py` (~200 lines pure Python) → `roofline.rs`. Predicts runtime in ~5µs CPU-only, replacing ~100ms `GemvAutotune` benchmarking.

Super-GOAT potential: If calibration is fast enough for per-game-session KV compression → "Personalized KV compression per combat encounter."

---

## Tasks

- [x] T1: Add `gram.wgsl` — compute X·Xᵀ (seq_len × seq_len Gram matrix) in WGSL
- [x] T2: Add dual-Gram routing to `spectralquant/calibration.rs` — dispatch based on `seq_len < 4 * d_h`
- [x] T3: GOAT proof — calibration accuracy with dual-Gram must match standard cov path
- [x] T4: Port `roofline.py` → `roofline.rs` in katgpt-core (~200 lines, pure Rust)
- [x] T5: GOAT proof — roofline predicted vs actual within ±20%
- [x] T6: Feature gate `dual_gram_pca` + `roofline_cost`, ensure OFF = zero impact

---

## T1: Gram Computation — CPU SIMD + GPU Kernel

Compute the Gram matrix G = X·Xᵀ where X is (seq_len × d_h).

### CPU SIMD path (seq_len ≤ 64)

For small Gram matrices, GPU launch overhead (~50µs) dominates computation (~1-2µs).
CPU SIMD wins for the very cases dual-Gram routing targets (short game sequences).

- `simd_gram_f32()` — chunked dot products, 4 or 8 elements at a time
- Output: `[f32; seq_len * seq_len]` on stack or pre-allocated scratch buffer
- Pre-allocate scratch buffer in calibration context, `clear()` + reuse across (layer, head) pairs
- Zero allocation inside the hot calibration loop

### GPU path (seq_len > 64)

- `gram.wgsl` — 2D grid, tile size 16×16
- Only compute upper triangle, mirror to lower
- Batch across (layer, head) pairs to amortize launch overhead

### Routing
```rust
if seq_len <= 64 {
    simd_gram_f32(&activation, seq_len, d_h, &mut gram_buf)  // CPU, ~1-5µs
} else {
    gram::compute_gram_gpu(&activation, seq_len, d_h, &mut gram_buf, &encoder)?  // GPU, amortized
}
```

### Why the floor at 64

GPU launch overhead: ~50µs. CPU 64×64×d_h Gram: ~5µs (d_h=128) to ~10µs (d_h=256).
At seq_len=64 the crossover is marginal. At seq_len=128, CPU cost ~20-40µs approaches GPU overhead.
Floor of 64 keeps CPU always winning; benchmark at implementation to confirm.

FlashLib reference: `primitives/pca/triton/pca.py` L73-116

---

## T2: Dual-Gram Routing in Calibration

Current code (`spectralquant/calibration.rs` L289) always computes d_h × d_h covariance.

Add routing:
```rust
if seq_len < 4 * d_h {
    // Dual-Gram path: compute X·Xᵀ (seq_len × seq_len), eigendecompose, project back
    gram::compute_gram(&activation, seq_len, d_h, &mut gram_buf, &encoder)?;
    // eigendecompose (seq_len × seq_len) — up to 512× smaller
    // Project eigenvectors: V = Xᵀ·U·Σ⁻¹
} else {
    // Standard covariance path: Xᵀ·X (d_h × d_h) — existing code
}
```

FlashLib crossover threshold: `N >= 4 * D` → cov path. Same heuristic applies.

---

## T3: GOAT Proof — Dual-Gram Calibration Accuracy

### Correctness

- Run calibration with standard cov path → record eigenvalues + eigenvectors
- Run calibration with dual-Gram path → record eigenvalues + eigenvectors
- Assert: max eigenvalue difference < 1e-4
- Assert: max eigenvector cosine distance < 1e-3
- Test on: d_h=128, d_h=256 with seq_len ∈ {16, 32, 64, 128, 256}
- Feature gate OFF must produce identical results to current code

### Performance

Follow profiling template from `.contexts/optimization.md`:
- Warmup: 100+ iterations
- Measure: 10,000+ iterations
- Use `std::hint::black_box()` on outputs
- Compare same-commit, back-to-back: OFF vs ON
- Print component-level breakdowns with `--nocapture`

Assert: dual-Gram path (CPU SIMD for seq_len≤64) must be faster than cov path for all seq_len < 4*d_h.
If not, raise the CPU floor or adjust crossover threshold.

### Binary bloat

- Isolate `dual_gram_pca` benchmarks into separate test file
- Compare no-feature vs with-feature binary size
- If regression appears only with feature enabled and code is properly gated → binary bloat, not a bug

---

## T4: `roofline.rs` — Cost Prediction

Port FlashLib's `info/roofline.py` (L269-327) to Rust:

```rust
pub struct RooflineCost {
    pub runtime_ms: f64,
    pub flops: u64,
    pub bytes_moved: u64,
    pub bound: ComputeBound,  // Compute | Memory | Launch
}

pub fn roofline_estimate(op: OpType, dtype: Dtype, flops: u64, bytes: u64) -> RooflineCost
```

- Calibrated throughput table: Apple M1/M2/M3/M4 peaks (from existing benchmarks)
- No GPU import — pure CPU calculation, ~5µs
- Unified cost surface for SR²AM/SpecHop/MaxSim dispatch decisions

FlashLib reference: `info/roofline.py` L269-327

---

## T5: GOAT Proof — Roofline Accuracy

### Accuracy

- For each (m, n) shape in existing GemvAutotune cache:
  - Predict with roofline model
  - Compare to actual benchmark time
- Assert: |predicted - actual| / actual < 0.20 (±20%)
- If not within tolerance → calibrate hardware peaks from benchmark data
- Feature gate OFF must produce identical behavior (no change to dispatch)

### Performance

Follow profiling template from `.contexts/optimization.md`:
- Warmup: 100+, Measure: 10,000+
- Verify roofline prediction call is ≤5µs (CPU-only, no GPU)
- Compare same-commit: roofline dispatch vs GemvAutotune dispatch

### Binary bloat

- Isolate `roofline_cost` benchmarks into separate test file
- Compare no-feature vs with-feature binary size

---

## T6: Feature Gates

- `dual_gram_pca` — default OFF, ON after T3 GOAT pass
- `roofline_cost` — default OFF, ON after T5 GOAT pass
- Both gates: OFF = zero code change (cfg guards on new paths only)

---

## Open/Close Boundary

- **katgpt-rs (open):** `RooflineCost` trait, `DualGramPca` trait, generic dispatch logic
- **riir-ai (private):** Game-specific calibration thresholds per domain, per-game seq_len/d_h crossover ratios
