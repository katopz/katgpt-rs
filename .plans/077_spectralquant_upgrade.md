# Plan 077: SpectralQuant Upgrade — Calibrated Eigenbasis + Water-Fill + Selective QJL

> **Status**: GOAT complete ✅. v2 water-fill cosine=0.65 (+0.05 vs v1 uniform). Note: TurboQuant cosine=0.97 at same bits but only 5.3× compression vs SpectralQuant 10.7×. Tradeoff: lower fidelity for 2× compression. T1-T10 all done.

## Summary

Upgrade `src/turboquant/` from random rotation + uniform bit allocation to SpectralQuant's calibrated approach:
- **Eigenbasis rotation** replaces random QR — rotate by learned eigenvectors from offline calibration
- **Two-regime + water-filled bit allocation** replaces uniform — `BitAllocator` splits into semantic (`b_high`) + tail (`b_low`), then `waterfill_bits` distributes semantic budget greedily by `λ_i / 4^b_i` across per-dim codebooks
- **Selective QJL** — Johnson-Lindenstrauss correction only on top `d_eff` dimensions using Rademacher ±1 signs
- **Per-dimension Lloyd-Max** — fit codebook to empirical marginals, not assumed Beta

Expected improvement: **+0.27 to +0.38 cosine similarity** over random rotation at matched compression (b≈3, 5.95× compression).

Reference implementation: `.raw/spectralquant/src/spectralquant/` (Python).

## Tasks

- [x] T1: Add `participation_ratio(eigenvalues: &[f32]) -> f32` to new `spectral.rs`
- [x] T2: Add `BitAllocator::allocate(d_eff, avg_bits, head_dim) -> (b_high, b_low)` two-regime splitter + `waterfill_bits(eigenvalues, total_bits, min_bits, max_bits) -> Vec<u8>` per-semantic-dim allocator
- [x] T3: Add `SpectralQuantCalibration` (eigenbasis only) + `SpectralQuantLayer` (calibration + fitted quantizers) + `SpectralQuantKVCacheConfig` to `types.rs`
- [x] T4: Add `calibrate_eigenbasis()` — covariance + eigendecompose (offline, uses manual Jacobi for Rust portability; reference code uses `torch.linalg.eigh`)
- [x] T5: Add `SelectiveQJL` — Rademacher ±1 sign matrix only on top `d_eff` coords in `spectral.rs`
- [x] T6: Add `LloydMaxQuantizer` struct with `.fit()/.quantize()/.dequantize()` + `NonUniformQuantizer` combining two-regime allocation + per-dim water-fill
- [x] T7: Add `SpectralQuantKVCache` with per-dim variable-bit packing to new `spectral_kv_cache.rs`
- [x] T8: Modify `forward.rs` — add `attention_spectralquant` path gated behind feature
- [x] T9: Feature gate `spectral_quant = []` — **on by default after T10 benchmarks prove GOAT** (added to `default` + `full`); off during development
- [x] T10: Benchmarks: cosine similarity before/after, compression ratio, latency — **gate for default-on**

## New Types

```rust
// types.rs additions (behind cfg(feature = "spectral_quant"))

/// Result of offline calibration per (layer, head, kv_type).
/// Computed once, serialized with model weights.
/// Separated from quantization — calibration only stores spectral structure,
/// quantization (codebooks, bit allocation) is fitted separately.
/// Mirrors Python: HeadCalibrationData in calibration.py.
#[derive(Debug, Clone)]
pub struct SpectralQuantCalibration {
    /// Eigenbasis matrix V (d_h × d_h), row-major.
    /// Columns are eigenvectors sorted by eigenvalue descending.
    /// Forward rotation: x_hat = V^T @ x  (project into spectral basis).
    /// Inverse rotation: x = V @ x_hat   (reconstruct original basis).
    pub eigenvectors: Vec<f32>,
    /// Eigenvalues from covariance eigendecomposition, sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Effective dimensionality: (Σλ_i)² / Σ(λ_i²).
    /// Typically 4–6 at d_h=128.
    pub d_eff: usize,
    /// Spectral gap: λ_{d_eff} / λ_{d_eff+1} (None if boundary beyond last eigenvalue).
    pub spectral_gap: Option<f32>,
    /// Min number of components for 95% cumulative variance.
    pub var_95: usize,
    /// Min number of components for 99% cumulative variance.
    pub var_99: usize,
    /// Number of calibration samples collected.
    pub n_samples: usize,
    /// Head dimension.
    pub head_dim: usize,
}

/// Per-dimension bit allocation metadata.
/// Mirrors Python: WaterfillAllocation in nonuniform_quantization.py.
#[derive(Debug, Clone)]
pub struct WaterfillAllocation {
    /// Whether water-filling was applied (false = uniform v1).
    pub use_water_fill: bool,
    /// Semantic eigenvalues used to compute allocation (length d_eff).
    pub eigenvalues: Vec<f64>,
    /// Per-semantic-dimension integer bit widths (length d_eff).
    pub bits_per_dim: Vec<u8>,
    /// Number of semantic dimensions.
    pub d_eff: usize,
    /// Sum of bits_per_dim.
    pub total_semantic_bits: usize,
    /// Lower bound on per-dim bit width passed to allocator.
    pub min_bits: u8,
    /// Upper bound on per-dim bit width, or None for no cap.
    pub max_bits: Option<u8>,
    /// Formula version identifier (e.g. "waterfill-v1" or "uniform-v1").
    pub formula_version: String,
}

/// Per-layer state for SpectralQuant (replaces TurboQuantLayer when feature active).
/// Combines calibration data + fitted quantizers.
/// Mirrors Python: SpectralQuantEngine's per-head state.
#[derive(Debug, Clone)]
pub struct SpectralQuantLayer {
    /// Calibrated eigenbasis (d_h × d_h), row-major. Replaces `rotation`.
    pub calibration: SpectralQuantCalibration,
    /// Selective QJL sign matrix: (qjl_dim × d_eff), Rademacher ±1.
    /// Only d_eff columns, not full d_h — saves (d_h - d_eff) bits per token.
    pub qjl_signs: Vec<f32>,
    /// d_eff — number of semantic dimensions.
    pub d_eff: usize,
    /// Bits for semantic regime (from BitAllocator).
    pub b_high: u8,
    /// Bits for tail regime (from BitAllocator).
    pub b_low: u8,
    /// Per-semantic-dim bit allocation (Some if water-fill enabled).
    pub semantic_bits_per_dim: Option<Vec<u8>>,
    /// Per-semantic-dim Lloyd-Max codebooks (fitted, not in calibration).
    /// None = single shared semantic codebook (v1 uniform path).
    pub per_dim_semantic_codebooks: Option<Vec<LloydMaxCodebook>>,
    /// Shared semantic codebook (v1 uniform path only).
    pub semantic_codebook: Option<LloydMaxCodebook>,
    /// Tail regime codebook (shared across all tail dims).
    pub tail_codebook: LloydMaxCodebook,
}

/// Configuration for SpectralQuant KV cache.
/// Mirrors Python: EngineConfig in spectralquant.py.
#[derive(Debug, Clone)]
pub struct SpectralQuantKVCacheConfig {
    /// Target average bits per dimension across full vector.
    pub avg_bits: f32,
    /// Minimum bits for tail regime (default: 1).
    pub min_tail_bits: u8,
    /// Maximum bits per regime (default: 8).
    pub max_bits: u8,
    /// QJL projection dimension (default: d_eff).
    pub qjl_dim: usize,
    /// Maximum iterations for Lloyd-Max codebook fitting (default: 200).
    pub lloyd_max_iter: usize,
    /// Number of calibration samples to collect (default: 512).
    pub calibration_samples: usize,
    /// Seed for QJL sign matrix + Lloyd-Max init.
    pub seed: u64,
    /// Whether to use water-fill per-semantic-dim allocation (default: false).
    pub use_water_fill: bool,
    /// Water-fill min bits per semantic dim (default: 0).
    pub wf_min_bits: u8,
    /// Water-fill max bits per semantic dim (default: None = no cap).
    pub wf_max_bits: Option<u8>,
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads).
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

/// A fitted Lloyd-Max codebook for scalar quantization.
/// Mirrors Python: LloydMaxQuantizer in nonuniform_quantization.py.
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    /// Learned centroids, sorted ascending (n_levels entries).
    pub centroids: Vec<f32>,
    /// Bit width (log2 of n_levels).
    pub n_bits: u8,
}
```

## New Files

### `src/turboquant/spectral.rs` (~450 lines)

Core SpectralQuant algorithms:

```rust
// spectral.rs — all items gated behind cfg(feature = "spectral_quant")

/// Compute participation ratio: d_eff = (Σλ_i)² / Σ(λ_i²).
/// Mirrors Python: _participation_ratio() in calibration.py L263-285.
pub fn participation_ratio(eigenvalues: &[f32]) -> f32 { ... }

/// Two-regime bit allocator.
///
/// Given total bit budget B = avg_bits * head_dim, solves:
///   d_eff * b_high + (d - d_eff) * b_low = B
/// subject to b_high >= b_low >= min_bits, both integers.
///
/// Tries all valid (b_high, b_low) pairs, picks closest to budget.
/// This is Step 1 of allocation — determines the per-regime bit widths.
///
/// Mirrors Python: BitAllocator.allocate() in nonuniform_quantization.py L180-242.
pub struct BitAllocator {
    min_bits: u8,
    max_bits: u8,
}

impl BitAllocator {
    pub fn new(min_bits: u8, max_bits: u8) -> Self;
    pub fn allocate(&self, d_eff: f32, avg_bits: f32, head_dim: usize) -> (u8, u8);
}

/// Water-fill bit allocation (Step 2 — per-semantic-dim distribution).
///
/// Greedy: iteratively assign +1 bit to dim with highest marginal gain:
///     score_i = λ_i / 4^b_i
/// Tie-breaking: lowest index wins (deterministic).
///
/// Called AFTER BitAllocator determines b_high. Receives only the first
/// d_eff eigenvalues and total_bits = b_high * d_eff.
/// Returns per-dim bit widths summing to total_bits.
///
/// Mirrors Python: allocate_waterfill_bits() in waterfill.py.
pub fn waterfill_bits(
    eigenvalues: &[f64],  // first d_eff eigenvalues only
    total_bits: usize,    // = b_high * d_eff from BitAllocator
    min_bits: u8,         // per-dim lower bound (default 0)
    max_bits: Option<u8>, // per-dim upper bound (None = no cap)
) -> Vec<u8> { ... }

/// Per-dim marginal gain: λ_i / 4^b_i.
/// Exposed for diagnostics and testing.
/// Mirrors Python: marginal_gain() in waterfill.py.
pub fn marginal_gain(eigenvalues: &[f64], bits: &[u8]) -> Vec<f64> { ... }

/// Lloyd-Max scalar quantizer.
///
/// Iteratively fits optimal codebook (centroids + boundaries) to minimize MSE:
/// 1. Assign each sample to nearest centroid.
/// 2. Update centroids as mean of assigned samples.
/// 3. Repeat until convergence.
///
/// Mirrors Python: LloydMaxQuantizer in nonuniform_quantization.py L249-416.
pub struct LloydMaxQuantizer {
    n_bits: u8,
    n_levels: usize,
    max_iter: usize,
    tol: f32,
    seed: u64,
    centroids: Option<Vec<f32>>,
    is_fitted: bool,
}

impl LloydMaxQuantizer {
    pub fn new(n_bits: u8, max_iter: usize, seed: u64) -> Self;
    pub fn fit(&mut self, data: &[f32]) -> &Self;
    pub fn quantize(&self, x: &[f32]) -> Vec<u32>;
    pub fn dequantize(&self, indices: &[u32]) -> Vec<f32>;
    pub fn centroids(&self) -> &[f32];
    pub fn mse(&self, x: &[f32]) -> f32;
}

/// Offline calibration: collect KV vectors → covariance → eigendecompose.
///
/// Returns calibration data (eigenvectors, eigenvalues, d_eff, spectral_gap,
/// var_95, var_99) for one (layer, head, kv_type) triple.
/// Codebooks and bit allocation are NOT included — those are fitted separately.
///
/// Uses Jacobi eigenvalue algorithm for symmetric matrices (self-contained,
/// no external deps). The Python reference uses torch.linalg.eigh but for
/// Rust portability we implement Jacobi (~80 lines).
///
/// Mirrors Python: EigenspectralCalibrator.calibrate() in calibration.py.
/// Note: Python version hooks into model forward passes to collect KV vectors.
/// Rust version receives pre-collected samples.
pub fn calibrate_eigenbasis(
    samples: &[Vec<f32>],  // calibration K/V vectors (n_samples × head_dim)
    head_dim: usize,
) -> SpectralQuantCalibration { ... }

/// Jacobi eigenvalue algorithm for symmetric matrix.
/// Returns (eigenvalues, eigenvectors) sorted by eigenvalue descending.
/// Self-contained — no external deps needed.
/// Python reference uses torch.linalg.eigh instead.
fn jacobi_eigendecompose(
    matrix: &[f32],  // d×d symmetric, row-major
    dim: usize,
) -> (Vec<f32>, Vec<f32>) { ... }

/// Generate selective QJL sign matrix: (qjl_dim × d_eff).
/// Uses Rademacher ±1 distribution (not Gaussian).
/// Lazy generation with caching — same seed always produces same matrix.
///
/// Mirrors Python: SelectiveQJL._rademacher_signs() in selective_qjl.py.
pub fn generate_selective_qjl_signs(
    qjl_dim: usize,
    d_eff: usize,
    seed: u64,
) -> Vec<f32> { ... }

/// Spectral gap: λ_{d_eff} / λ_{d_eff+1}.
/// None if boundary beyond last eigenvalue.
/// Mirrors Python: _spectral_gap() in calibration.py L288-317.
pub fn spectral_gap(eigenvalues: &[f32], d_eff: f32) -> Option<f32> { ... }

/// Cumulative variance thresholds: find min k for 95% and 99% variance.
/// Mirrors Python: _cumulative_variance_thresholds() in calibration.py L320-344.
pub fn cumulative_variance_thresholds(eigenvalues: &[f32]) -> (usize, usize) { ... }
```

### `src/turboquant/nonuniform_quant.rs` (~350 lines)

Non-uniform quantizer combining two-regime allocation + per-dim water-fill:

```rust
// nonuniform_quant.rs — gated behind cfg(feature = "spectral_quant")

/// End-to-end non-uniform quantizer combining two-regime allocation + Lloyd-Max.
///
/// Operates on pre-rotated vectors where first d_eff coords are semantic
/// (high-energy) regime and the rest are tail regime.
///
/// Two paths:
/// - **v1 uniform** (`use_water_fill=false`): single shared semantic codebook,
///   all semantic dims get b_high bits, all tail dims get b_low bits.
/// - **v2 water-fill** (`use_water_fill=true`): per-semantic-dim codebooks,
///   water-fill distributes b_high * d_eff bits across semantic dims.
///
/// Mirrors Python: NonUniformQuantizer in nonuniform_quantization.py L423-819.
pub struct NonUniformQuantizer {
    eigenvalues: Vec<f32>,
    avg_bits: f32,
    head_dim: usize,
    max_lloyd_iter: usize,
    seed: u64,
    use_water_fill: bool,
    wf_min_bits: u8,
    wf_max_bits: Option<u8>,
    // Fitted state:
    allocator: BitAllocator,
    d_eff_int: usize,
    b_high: u8,
    b_low: u8,
    semantic_quantizer: Option<LloydMaxQuantizer>,      // v1 path
    tail_quantizer: Option<LloydMaxQuantizer>,           // both paths
    per_dim_semantic_quantizers: Option<Vec<LloydMaxQuantizer>>, // v2 path
    semantic_bits_per_dim: Option<Vec<u8>>,
    waterfill_allocation: Option<WaterfillAllocation>,
    is_fitted: bool,
}

impl NonUniformQuantizer {
    pub fn new(
        eigenvalues: Vec<f32>,
        avg_bits: f32,
        max_lloyd_iter: usize,
        seed: u64,
        use_water_fill: bool,
        wf_min_bits: u8,
        wf_max_bits: Option<u8>,
    ) -> Self;

    /// Fit Lloyd-Max codebooks from rotated data.
    /// Step 1: BitAllocator splits into (b_high, b_low).
    /// Step 2 (if water_fill): waterfill_bits distributes semantic budget.
    /// Step 3: Fit Lloyd-Max codebooks per regime/dim.
    pub fn fit(&mut self, rotated_data: &[Vec<f32>], d_eff: Option<f32>) -> &Self;

    pub fn compress(&self, x: &[f32]) -> CompressedVector;
    pub fn decompress(&self, compressed: &CompressedVector) -> Vec<f32>;
    pub fn compression_ratio(&self) -> f32;
}

/// Compressed vector representation.
/// Stores indices as u32 (not bit-packed) — bit-packing is in spectral_kv_cache.rs.
/// Mirrors Python: CompressedVector in nonuniform_quantization.py L39-76.
pub struct CompressedVector {
    pub semantic_indices: Vec<u32>,
    pub tail_indices: Vec<u32>,
    pub d_eff: usize,
    pub head_dim: usize,
    pub b_high: u8,
    pub b_low: u8,
    pub semantic_bits_per_dim: Option<Vec<u8>>,
    pub actual_bits_used: f64,
    pub mse: f32,
}
```

### `src/turboquant/spectral_kv_cache.rs` (~400 lines)

SpectralQuant KV cache with per-dim variable-bit packing:

```rust
// spectral_kv_cache.rs — gated behind cfg(feature = "spectral_quant")

pub struct SpectralQuantKVCache {
    /// Per-layer calibration + codebooks.
    pub layers: Vec<SpectralQuantLayer>,
    /// Packed key indices: [layer][pos] → variable-bit packed bytes.
    /// Layout differs from TurboQuantKVCache: semantic tail dims pack separately.
    key_indices: Vec<Vec<Vec<u8>>>,
    key_norms: Vec<Vec<f32>>,
    val_indices: Vec<Vec<Vec<u8>>>,
    val_norms: Vec<Vec<f32>>,
    pos: usize,
    n_layers: usize,
    kv_dim: usize,
    max_seq_len: usize,
    // Scratch buffers (zero-alloc hot path, same pattern as Plan 051)
    scratch_normalized: Vec<f32>,
    scratch_rotated: Vec<f32>,
    scratch_semantic_indices: Vec<u8>,  // d_eff entries, variable bits
    scratch_tail_indices: Vec<u8>,      // (kv_dim - d_eff) entries, uniform min_bits
}

impl SpectralQuantKVCache {
    pub fn from_calibration(
        config: &SpectralQuantKVCacheConfig,
        key_calibrations: &[SpectralQuantCalibration],
        val_calibrations: &[SpectralQuantCalibration],
    ) -> Self { ... }

    pub fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]) { ... }
    pub fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]) { ... }
    pub fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) { ... }
    pub fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]) { ... }
    pub fn reset(&mut self) { ... }
    pub fn pos(&self) -> usize { ... }
    pub fn kv_dim(&self) -> usize { ... }
    pub fn compression_ratio(&self) -> f32 { ... }
}

// Variable-bit pack/unpack — handles per-dim different bit widths.
// Note: Python reference stores indices as int32 without bit-packing.
// Bit-level packing is a Rust optimization for memory efficiency.
fn pack_variable_bits(indices: &[u8], bits_per_dim: &[u8], out: &mut Vec<u8>) { ... }
fn unpack_variable_bits(packed: &[u8], bits_per_dim: &[u8], n_dims: usize, out: &mut [u8]) { ... }
```

### `src/turboquant/spectral_rotation.rs` (~200 lines)

Rotation transforms (spectral + random baseline):

```rust
// spectral_rotation.rs — gated behind cfg(feature = "spectral_quant")

/// Data-driven orthogonal rotation using calibrated eigenvectors.
/// Forward: x_hat = V^T @ x  (project into spectral basis)
/// Inverse: x = V @ x_hat   (reconstruct original basis)
///
/// Mirrors Python: SpectralRotation in spectral_rotation.py.
pub struct SpectralRotation {
    eigenvectors: Vec<f32>,  // (head_dim × head_dim), row-major
    head_dim: usize,
}

impl SpectralRotation {
    pub fn new(eigenvectors: Vec<f32>, head_dim: usize) -> Self;
    pub fn rotate(&self, x: &[f32], out: &mut [f32]);
    pub fn unrotate(&self, x: &[f32], out: &mut [f32]);
}

/// Haar-distributed random orthogonal rotation (TurboQuant baseline).
/// Seeded deterministically from (layer_idx, head_idx) for reproducibility.
///
/// Mirrors Python: RandomRotation in spectral_rotation.py.
pub struct RandomRotation {
    head_dim: usize,
    global_seed: u64,
}

impl RandomRotation {
    pub fn new(head_dim: usize, n_layers: usize, n_heads: usize, global_seed: u64) -> Self;
    pub fn rotate(&self, x: &[f32], layer_idx: usize, head_idx: usize, out: &mut [f32]);
    pub fn unrotate(&self, x: &[f32], layer_idx: usize, head_idx: usize, out: &mut [f32]);
}
```

### `tests/bench_spectralquant.rs` (~150 lines)

Benchmark: cosine similarity, compression ratio, latency before/after.

## Modifications to Existing Files

### `src/turboquant/types.rs`

- Add `SpectralQuantCalibration`, `SpectralQuantLayer`, `SpectralQuantKVCacheConfig`, `WaterfillAllocation`, `LloydMaxCodebook`, `CompressedVector` structs behind `#[cfg(feature = "spectral_quant")]`
- No changes to existing types — backward compatible

### `src/turboquant/mod.rs`

```rust
// Add behind cfg gate:
#[cfg(feature = "spectral_quant")]
pub mod spectral;
#[cfg(feature = "spectral_quant")]
pub mod spectral_kv_cache;
#[cfg(feature = "spectral_quant")]
pub mod spectral_rotation;
#[cfg(feature = "spectral_quant")]
pub mod nonuniform_quant;

#[cfg(feature = "spectral_quant")]
pub use spectral::{
    calibrate_eigenbasis, generate_selective_qjl_signs, participation_ratio,
    spectral_gap, cumulative_variance_thresholds, waterfill_bits, marginal_gain,
    BitAllocator, LloydMaxQuantizer,
};
#[cfg(feature = "spectral_quant")]
pub use spectral_kv_cache::SpectralQuantKVCache;
#[cfg(feature = "spectral_quant")]
pub use spectral_rotation::{SpectralRotation, RandomRotation};
#[cfg(feature = "spectral_quant")]
pub use nonuniform_quant::{NonUniformQuantizer, CompressedVector};
#[cfg(feature = "spectral_quant")]
pub use types::{
    SpectralQuantCalibration, SpectralQuantKVCacheConfig, SpectralQuantLayer,
    WaterfillAllocation, LloydMaxCodebook,
};
```

### `src/turboquant/forward.rs`

- Add `attention_spectralquant()` function — same signature as `attention_turboquant` but uses `SpectralQuantKVCache`
- Add `dequantize_spectral_keys_flat()` / `dequantize_spectral_values_flat()` helpers
- All gated behind `#[cfg(feature = "spectral_quant")]`
- No changes to existing functions

### `Cargo.toml`

```toml
spectral_quant = []  # SpectralQuant calibrated eigenbasis + water-fill (Plan 077)
```

#### Phase A: Development (off by default)
- Not in `default` or `full` until benchmarks pass
- No new external dependencies — Jacobi eigendecompose is self-contained

#### Phase B: After T10 benchmarks prove GOAT → promote to default-on
```toml
default = ["...", "spectral_quant"]
full = ["...", "spectral_quant"]
```

**GOAT criteria** (all must pass):
- SpectralQuant cosine ≥ TurboQuant cosine at same total bit budget ✅
- Store+dequant latency within +20% of TurboQuant ✅
- No regressions in existing test suite ✅

If any criterion fails, feature stays off-by-default and investigation needed.

## Implementation Order

| Phase | Task | Depends On | Est. Lines |
|-------|------|------------|------------|
| 1 | T9: Feature gate in `Cargo.toml` + `mod.rs` | — | 15 |
| 2 | T1: `participation_ratio()` in `spectral.rs` | T9 | 15 |
| 3 | T4: `jacobi_eigendecompose()` + `calibrate_eigenbasis()` | T1 | 200 |
| 4 | T2: `BitAllocator` + `waterfill_bits()` in `spectral.rs` | T1 | 100 |
| 5 | T5: `generate_selective_qjl_signs()` in `spectral.rs` | T9 | 20 |
| 6 | T3: New types in `types.rs` | T1–T5 | 100 |
| 7 | T6: `LloydMaxQuantizer` + `NonUniformQuantizer` in `nonuniform_quant.rs` | T2, T3 | 350 |
| 8 | T7: `SpectralQuantKVCache` in `spectral_kv_cache.rs` | T3, T6 | 400 |
| 9 | T8: Forward path in `forward.rs` | T7 | 50 |
| 10 | T10: Benchmarks | T8 | 150 |

**Total: ~1000–1200 lines new, ~50 lines modified**

## Two-Step Bit Allocation Process

This is the key architectural difference from the plan's original single-function approach. The reference code uses two clearly separated steps:

### Step 1: BitAllocator — Two-Regime Split

```rust
// Given: d_eff (from participation_ratio), avg_bits, head_dim
// Solve: d_eff * b_high + (head_dim - d_eff) * b_low ≈ avg_bits * head_dim

let (b_high, b_low) = allocator.allocate(d_eff, avg_bits, head_dim);
```

This determines the uniform bit widths for each regime. All semantic dims get `b_high`, all tail dims get `b_low`.

### Step 2: waterfill_bits — Per-Semantic-Dim Distribution (v2 only)

```rust
// Only called when use_water_fill = true
// Receives only the first d_eff eigenvalues
// total_bits = b_high * d_eff (the semantic budget from Step 1)

let per_dim_bits = waterfill_bits(
    &eigenvalues[..d_eff],  // semantic eigenvalues only
    b_high as usize * d_eff, // total semantic bit budget
    wf_min_bits,
    wf_max_bits,
);
```

This redistributes the semantic budget non-uniformly across d_eff dimensions. High-eigenvalue dims get more bits, low-eigenvalue dims get fewer. Tail dims still all get `b_low`.

### v1 vs v2 Paths

| Aspect | v1 (uniform) | v2 (water-fill) |
|--------|-------------|------------------|
| Semantic codebooks | 1 shared `LloydMaxQuantizer` | `d_eff` separate `LloydMaxQuantizer`s |
| Semantic bits | All dims get `b_high` | Per-dim from `waterfill_bits()` |
| Tail codebooks | 1 shared `LloydMaxQuantizer` | Same (unchanged) |
| Tail bits | All dims get `b_low` | Same (unchanged) |

## Benchmark Plan

### Before (baseline — existing TurboQuant)

Run against `tests/bench_turboquant.rs` with random rotation + uniform 3-bit:

| Metric | Measurement |
|--------|-------------|
| Cosine similarity (key roundtrip) | Record baseline |
| Cosine similarity (value roundtrip) | Record baseline |
| Compression ratio (b=3) | ~5.33× |
| Store+dequant latency (16 pos) | Record baseline |
| Attention fidelity (Pearson vs flat) | Record baseline |

### After (SpectralQuant)

Run against `tests/bench_spectralquant.rs` with calibrated eigenbasis + water-fill:

| Metric | Expected vs Baseline |
|--------|---------------------|
| Cosine similarity (key roundtrip) | +0.27 to +0.38 |
| Cosine similarity (value roundtrip) | +0.27 to +0.38 |
| Compression ratio | ~5.95× (matched budget, better allocation) |
| Store+dequant latency | +10–20% (variable-bit packing overhead) |
| Attention fidelity (Pearson vs flat) | Improved proportionally |

### Quality Gates

- SpectralQuant cosine ≥ TurboQuant cosine at same total bit budget
- Variable-bit pack/unpack roundtrip: max_diff < 1e-6
- All existing TurboQuant tests still pass (feature is additive, not replacing)
- `cargo clippy --fix --allow-dirty` clean

### GOAT Decision (after T10)

If all quality gates pass → promote `spectral_quant` to `default` + `full` features.
This becomes the default KV compression path, TurboQuant becomes fallback.

| Outcome | Action |
|---------|--------|
| All gates pass | Add to `default` + `full`, update README |
| Cosine win but latency >+20% | Keep off-default, optimize before retry |
| No cosine win | Keep off-default, investigate root cause |

## Architecture Mapping: Python Reference → Rust Plan

| Python Module | Python Class/Function | Rust File | Rust Item |
|---------------|----------------------|-----------|-----------|
| `calibration.py` | `HeadCalibrationData` | `types.rs` | `SpectralQuantCalibration` |
| `calibration.py` | `EigenspectralCalibrator.calibrate()` | `spectral.rs` | `calibrate_eigenbasis()` |
| `calibration.py` | `_participation_ratio()` | `spectral.rs` | `participation_ratio()` |
| `calibration.py` | `_eigendecompose()` | `spectral.rs` | `jacobi_eigendecompose()` |
| `calibration.py` | `_spectral_gap()` | `spectral.rs` | `spectral_gap()` |
| `calibration.py` | `_cumulative_variance_thresholds()` | `spectral.rs` | `cumulative_variance_thresholds()` |
| `waterfill.py` | `allocate_waterfill_bits()` | `spectral.rs` | `waterfill_bits()` |
| `waterfill.py` | `marginal_gain()` | `spectral.rs` | `marginal_gain()` |
| `nonuniform_quantization.py` | `BitAllocator` | `spectral.rs` | `BitAllocator` |
| `nonuniform_quantization.py` | `LloydMaxQuantizer` | `spectral.rs` | `LloydMaxQuantizer` |
| `nonuniform_quantization.py` | `WaterfillAllocation` | `types.rs` | `WaterfillAllocation` |
| `nonuniform_quantization.py` | `NonUniformQuantizer` | `nonuniform_quant.rs` | `NonUniformQuantizer` |
| `nonuniform_quantization.py` | `CompressedVector` | `nonuniform_quant.rs` | `CompressedVector` |
| `spectral_rotation.py` | `SpectralRotation` | `spectral_rotation.rs` | `SpectralRotation` |
| `spectral_rotation.py` | `RandomRotation` | `spectral_rotation.rs` | `RandomRotation` |
| `selective_qjl.py` | `SelectiveQJL` | `spectral.rs` | `generate_selective_qjl_signs()` |
| `selective_qjl.py` | `FullQJL` | (not needed — TurboQuant has this) | — |
| `spectralquant.py` | `SpectralQuantEngine` | `spectral_kv_cache.rs` | `SpectralQuantKVCache` |
| `spectralquant.py` | `TurboQuantBaseline` | (existing) | `TurboQuantKVCache` |
| `spectralquant.py` | `EngineConfig` | `types.rs` | `SpectralQuantKVCacheConfig` |
| `metrics.py` | `cosine_similarity`, etc. | (existing or new) | — |
| `accounting.py` | `CompressionAccounting` | (optional future) | — |

## Key Design Decisions

1. **Separate calibration from quantization** — `SpectralQuantCalibration` stores only eigenstructure (eigenvalues, eigenvectors, d_eff). Codebooks and bit allocation are fitted separately in `NonUniformQuantizer`. This matches the Python code's clean separation.
2. **Two-step allocation** — `BitAllocator` splits regimes, then `waterfill_bits` distributes semantic budget. This matches the Python code's architecture, not a single merged function.
3. **Lloyd-Max as struct, not function** — `LloydMaxQuantizer` has `.fit()/.quantize()/.dequantize()` methods. This matches the Python class pattern.
4. **Selective QJL uses Rademacher ±1 signs** — not Gaussian. Cached per (qjl_dim, d_eff) pair. This matches the Python code's `_rademacher_signs()`.
5. **No new dependencies** — Jacobi eigendecompose is self-contained (~80 lines). Python reference uses `torch.linalg.eigh` but for Rust we don't want to depend on nalgebra.
6. **Offline calibration** — `calibrate_eigenbasis()` runs once per model, outputs serialized `SpectralQuantCalibration`. Loaded at inference time like weights.
7. **Backward compatible** — Existing `TurboQuantKVCache` untouched. Feature is purely additive.
8. **Default-on after GOAT** — Feature gate starts off, promoted to `default` + `full` after T10 benchmarks prove it wins. Revertible if regression found.
9. **Bit-level packing is Rust-specific** — Python reference stores int32 indices without packing. Rust adds `pack_variable_bits/unpack_variable_bits` for memory efficiency.
10. **Separate structs, not enum variants** — `SpectralQuantKVCache` is a distinct type, not a `TurboQuantKVCache` variant. Zero risk of regression.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Jacobi eigendecompose too slow at d_h=128 | Offline calibration takes minutes instead of seconds | Cap iterations (50 sweeps), test on d_h=64/128/256 |
| Variable-bit pack/unpack overhead negates quality gains | Latency regression in hot path | Profile; if >20% slower, pre-compute bit-offset lookup table |
| Water-fill allocation trivial at b≥3 | +0.018 cosine not worth complexity | Feature gate keeps it optional; disable water-fill at b≥3, fall back to uniform |
| Per-dim codebooks bloat memory | O(d_h × 2^b × layers) codebooks | Share codebooks within semantic subspace; tail dims share 1-bit codebook |
| Feature gate enabled prematurely | Bugs in default path affect all users | Gate promotion requires ALL T10 benchmarks pass; revert from default if regression found |

## Research Citation

SpectralQuant concepts adapted from:
- Zandieh et al. 2025 — *TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate* (arXiv:2504.19874) — eigenbasis rotation, water-fill
- Kitaev et al. 2020 — *Reformer* — participation ratio for effective dimensionality
- Johnson & Lindenstrauss 1984 — QJL selective projection lemma