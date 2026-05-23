# Research 63: OCTOPUS — Octahedral KV Cache Compression

**Paper:** [arXiv:2605.21226](https://arxiv.org/abs/2605.21226) (May 2026)
**Authors:** Mark Boss, Vikram Voleti, Simon Donné, Shimon Vainer (Stability AI)
**Venue:** Preprint

## TL;DR

OCTOPUS quantizes KV cache coordinate **triplets** jointly via an octahedral map (S² → [-1,1]²), with MSE-optimal non-uniform bit allocation (b+1, b-1) for direction vs. norm. Beats TurboQuant and PolarQuant at every bit width across text, video, and audio. Data-oblivious, online, deterministic given seed.

## Core Innovation

### Triplet Decomposition (not per-coordinate)

TurboQuant quantizes each rotated coordinate independently. OCTOPUS groups rotated coordinates into contiguous triplets `t_i ∈ R³`, then decomposes each into:
- **Norm** `ρ_i = ||t_i||₂` — has Beta(3/2, (d-3)/2) marginal, concentrated near √(3/d) → fewer bits needed
- **Direction** `n_i = t_i/ρ_i ∈ S²` — encoded via octahedral map to two scalars on [-1,1]²

```text
k ∈ R^d → γ (fp32) + Rû (rotated unit direction)
         → n_tri = ⌈d/3⌉ triplets
         → each triplet: (ρ_i, ξ_i, η_i) via octahedral fold
         → Lloyd-Max quantize with (b+1, b-1) split
```

### Octahedral Map (from Computer Graphics)

Equal-area parameterization of S² → [-1,1]²:
- Piecewise-linear encode/decode in O(1)
- Constant Jacobian per octant
- Near-uniform Jacobian makes 1D Lloyd-Max a close approximation to true 2-sphere distortion

```text
Encode: (x,y,z) on S² → project to octahedron {ℓ=1} → unfold to square
  if z ≥ 0: (ξ, η) = (x/ℓ, y/ℓ)
  if z < 0:  (ξ, η) = (sign(x)(1-|y|), sign(y)(1-|x|))

Decode: (ξ, η) → reconstruct z = 1 - |ξ| - |η| → normalize
```

### MSE-Optimal Non-Uniform Bit Split

Key insight: direction errors dominate because `E[ρ²_i] = 3/d → 0`, while direction variance is O(1).

Lagrangian optimum yields: `b*_dir - b*_nrm = O(1)` (independent of d and total budget!)

At d=128, the optimal split is **(b+1, b-1)**:
- Total bits per triplet: `2(b+1) + (b-1) = 3b + 1` (vs. 3b uniform)
- MSE reduction: **31-41%** vs. uniform (b,b) split
- Verified empirically across all tested bit widths

### Joint 3×3 Rounding

Instead of independent nearest-centroid rounding for (ξ, η, ρ), OCTOPUS does local 3×3 search:
1. Start from scalar Lloyd seed (j_x, j_y)
2. Enumerate 9 candidates: {(j_x + δ_x, j_y + δ_y) | δ_x, δ_y ∈ {-1, 0, 1}}
3. Pick direction maximizing `s_i = t_i^T · n(ξ, η)` (dot product with true vector)
4. Pick norm nearest to `s_i` (NOT to ||t_i|| — key insight!)

Result: **6-14% MSE reduction** at zero bitstream change (encoder-only, same decoder).

### Optional QJL 1-bit Residual (OCTOPUS-QJL)

Composes with existing QJL technique:
- Store `σ = sign(R'r)` where R' is a second independent rotation
- Adds ~0.5 bits/scalar effective rate
- Drives dot-product bias → 0 (unbiased estimator)
- Only useful for score-attention deployments

## Benchmark Results (Key Numbers)

### Synthetic (d=128, Gaussian keys)
| bits | Codec | MSE | IP Error | 
|------|-------|-----|----------|
| 2 | TurboQuant | 0.1161 | 3.054 |
| 2 | **OCTOPUS** | **0.0897** | **2.682** |
| 4 | TurboQuant | 0.0094 | 0.866 |
| 4 | **OCTOPUS** | **0.0071** | **0.753** |

### Qwen2.5-7B-Instruct-1M (WikiText-2 PPL delta)
| bits | TurboQuant | PolarQuant | OCTOPUS |
|------|------------|------------|---------|
| 2 | +63.0% | +186.6% | **+34.7%** |
| 3 | +8.6% | +15.7% | **+7.2%** |
| 4 | +3.1% | +4.4% | **+2.7%** |

### Needle-in-a-Haystack (b=2, 4k-128k context)
- OCTOPUS: **0.81 recall** (only codec retaining usability)
- TurboQuant: 0.63 recall
- PolarQuant: **0.05 recall** (collapsed)

### Video (CausVid, b=2 LPIPS)
- TurboQuant-QJL: 0.579 mean (near noise)
- **OCTOPUS: 0.178 mean** (still coherent)

## Distillation to Our Architecture

### What We Already Have

| Component | Our Codebase | OCTOPUS Equivalent |
|-----------|-------------|-------------------|
| Rotation | `turboquant/rotation.rs` — WHT + random sign flips | Same (sign-flipped WHT) |
| Codebook | `turboquant/codebook.rs` — Lloyd-Max | Same algorithm, different marginals |
| KV Cache | `QuantizedKVCache` trait | Implement same trait |
| QJL Residual | `turboquant/rotation.rs` — `qjl_matrix` | Identical technique |
| Pack/Unpack | `turboquant/kv_cache.rs` — bit-packed indices | Same, with triplet grouping |
| GPU Path | `riir-gpu/spectralquant/` | Future: fused WGSL decode |

### What We Need to Add

1. **Octahedral map** — encode/decode S² ↔ [-1,1]² (new module)
2. **Triplet decomposition** — group rotated coords into 3-blocks (new)
3. **Non-uniform bit split** — (b+1, b-1) for direction/norm (new config)
4. **Triplet-norm codebook** — Beta(3/2, (d-3)/2) marginal (new marginal)
5. **Oct-coordinate codebook** — empirical marginal from octahedral fold (new marginal)
6. **Joint 3×3 rounding** — encoder-only optimization (new in encoder)
7. **Score-path decode** — never materialize K, compute score from packed state (new decode path)

### Architecture Mapping

```
microgpt-rs/src/
├── turboquant/          # Existing — legacy baseline
├── spectralquant/       # Existing — current default (eigenbasis + water-fill)
└── octopus/             # NEW — triplet octahedral codec
    ├── mod.rs           # Module index + re-exports
    ├── types.rs         # OctopusConfig, OctopusLayer, OctopusCodebook
    ├── octahedral.rs    # S² ↔ [-1,1]² encode/decode
    ├── codebook.rs      # Triplet-norm + oct-coordinate Lloyd-Max
    ├── rotation.rs      # Reuse from turboquant (same WHT)
    ├── encode.rs        # Triplet decomposition + joint 3×3 rounding
    ├── kv_cache.rs      # OctopusKVCache (impl QuantizedKVCache)
    └── forward.rs       # Score-path decode (no K materialization)

riir-ai/crates/riir-gpu/src/
└── octopus/             # NEW — GPU fused kernels
    ├── mod.rs
    ├── encode.rs        # WGSL triplet encode
    └── attention.rs     # Fused octahedral decode + attention
```

## Verdict

### Why OCTOPUS Over Current Options

| Aspect | TurboQuant | SpectralQuant | OCTOPUS |
|--------|------------|---------------|---------|
| Calibration | None (data-oblivious) | Requires data samples | None (data-oblivious) |
| Latency overhead | Low | Medium (eigenbasis) | Medium (triplet math) |
| 2-bit quality | Collapses (63% PPL) | Good (water-fill adapts) | Best (34.7% PPL) |
| 4-bit quality | Good | Very good | Best |
| Online | ✅ Yes | ✅ Yes (after calibration) | ✅ Yes |
| GPU-friendly | ✅ Simple | ⚠️ Complex kernels | ✅ Regular access patterns |
| Deterministic | ✅ Seed-based | ✅ Seed-based | ✅ Seed-based |

### Strategic Assessment

**OCTOPUS is the natural successor to TurboQuant in the rotation-preconditioned family.** It:
1. Shares the same WHT rotation (reuses existing code)
2. Is data-oblivious (no calibration overhead — advantage over SpectralQuant)
3. Has regular memory access patterns (triplet-aligned → GPU-friendly)
4. Dominates at 2-bit (critical for extreme compression scenarios)
5. The octahedral map is simple piecewise-linear math — Rust-friendly

**SpectralQuant remains the best choice when calibration data is available** because water-fill allocation adapts per-dimension. OCTOPUS wins when:
- No calibration data available (cold start)
- Need deterministic worst-case guarantees
- GPU deployment with simple kernels
- Multi-modal (video, audio — SpectralQuant is LLM-only calibrated)

### Recommended Position in Stack

```
Production Stack (GOAT):
  1. SpectralQuant  — default, when calibration available (highest quality)
  2. OCTOPUS        — fallback, data-oblivious, extreme compression (2-3 bit)
  3. TurboQuant     — legacy, kept for backward compatibility
```

### Feature Gate

```toml
[features]
default = ["spectral_quant"]
turboquant = []           # Legacy baseline
spectral_quant = []       # Current default (calibrated)
octopus = []              # NEW: data-oblivious triplet codec
sp_kv = []                # Composable with any above
```

### Risks & Limitations (from paper)

1. **Wall-clock overhead**: OCTOPUS encode/decode adds 5-11× vs. bf16 SDPA (Table 7). Acceptable when KV bandwidth/capacity is bottleneck, not when compute-bound.
2. **d must be power of 2**: Required by WHT. Already satisfied by standard transformer dims.
3. **(b+1, b-1) split verified at d=128**: May differ at d=64, d=256 — need sweep.
4. **QJL variant only for score-attention**: Not useful for reconstruction-path deployments.

## Implementation Complexity Estimate

| Component | LOC Estimate | Difficulty |
|-----------|-------------|------------|
| Octahedral map | ~80 lines | Low (piecewise linear) |
| Triplet decomposition | ~60 lines | Low (contiguous slicing) |
| Codebook (2 marginals) | ~100 lines | Medium (reuse Lloyd-Max) |
| Joint 3×3 rounding | ~80 lines | Medium |
| KV Cache struct | ~300 lines | Medium (follow TurboQuant pattern) |
| Score-path decode | ~120 lines | Medium |
| GOAT benchmark | ~200 lines | Low (existing framework) |
| GPU kernels (future) | ~400 lines | High (WGSL fused attention) |

**Total: ~940 lines CPU, ~400 lines GPU (future)**

## References

- OCTOPUS project page: https://octopus-quant.github.io/
- Octahedral map: Cigolle et al. "A Survey of Efficient Representations for Independent Unit Vectors" JCGT 2014
- TurboQuant: Zandieh et al. arXiv 2025 (our `turboquant/`)
- PolarQuant: Han et al. arXiv 2025 (recursive polar angles — alternative to octahedral)
- QJL: Zandieh et al. arXiv 2024 (our existing QJL in `turboquant/rotation.rs`)
- Lloyd-Max: Lloyd 1982, Max 1960 (our existing `codebook.rs`)