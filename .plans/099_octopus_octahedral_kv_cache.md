# Plan 099: OCTOPUS — Octahedral Triplet KV Cache Compression

> **Research:** 63 (OCTOPUS Octahedral KV Cache Compression)
> **Related Plans:** 043 (TurboQuant), 077 (SpectralQuant Upgrade), 044 (PFlash), 080 (MaxSim), 095 (GRAM Width/Depth GOAT), 096 (MoE+SD CoDesign GOAT), 050 (Feature Gate Audit)
**Status:** ✅ Complete
> **Verdict:** High-value addition. OCTOPUS is the first data-oblivious codec to beat calibrated SpectralQuant at all bit widths (-22% to -49% MSE, Bench 022). Data-oblivious (no calibration), regular memory access patterns (GPU-friendly), simple piecewise-linear math (Rust-friendly). Default-on as of Plan 099. SpectralQuant kept for per-dimension water-fill scenarios; TurboQuant demoted to legacy baseline.

## Summary

Implement the OCTOPUS codec (arXiv:2605.21226) for KV cache compression. Core innovations vs. our existing TurboQuant:

1. **Triplet decomposition** — groups rotated coordinates into contiguous 3-blocks instead of per-coordinate quantization
2. **Octahedral map** — S² → [-1,1]² equal-area parameterization from computer graphics
3. **Non-uniform bit split** — (b+1, b-1) for direction/norm is MSE-optimal at d=128 (31-41% MSE reduction)
4. **Joint 3×3 rounding** — encoder-only optimization (6-14% MSE reduction, zero decoder change)
5. **Score-path decode** — reconstruct keys on-the-fly from packed state without materializing K

### Key Numbers (from paper)

| bits | TurboQuant PPL Δ | OCTOPUS PPL Δ | Improvement |
|------|-------------------|---------------|-------------|
| 2    | +63.0%            | +34.7%        | 1.8× better |
| 3    | +8.6%             | +7.2%         | 1.2× better |
| 4    | +3.1%             | +2.7%         | 1.1× better |

At 2-bit, OCTOPUS is the only codec that doesn't collapse on needle-in-a-haystack (0.81 recall vs. 0.05 for PolarQuant).

## Tasks

- [x] T1: Add `octahedral` module — S² ↔ [-1,1]² encode/decode with unit tests
- [x] T2: Add `triplet` module — decomposition of rotated vector into (ρ, n) pairs
- [x] T3: Add `codebook` module — triplet-norm Beta marginal + oct-coordinate empirical marginal
- [x] T4: Add `octopus/types.rs` — `OctopusConfig`, `OctopusLayer`, `OctopusCodebook`
- [x] T5: Add `octopus/encode.rs` — triplet encode with joint 3×3 rounding
- [x] T6: Add `octopus/kv_cache.rs` — `OctopusKVCache` implementing `QuantizedKVCache` trait
- [x] T7: Add `octopus/forward.rs` — score-path decode + attention scoring helpers
- [x] T8: Add `octopus` feature gate to `Cargo.toml` + conditional module in `src/lib.rs`
- [x] T9: Add GOAT benchmark — synthetic MSE, cosine, IP error sweep (2/3/4 bit, d=64/128/256)
- [x] T10: Add GOAT benchmark — compression ratio comparison vs. TurboQuant at matched bits
- [x] T11: Run GOAT proof, record results in `.benchmarks/022_octopus_goat.md`
- [x] T12: Update `README.md` with OCTOPUS section + production stack ordering

## Architecture

### Module Structure

```
microgpt-rs/src/
├── types.rs                 # Existing: QuantizedKVCache trait (unchanged)
├── turboquant/              # Existing: legacy baseline (unchanged)
├── spectralquant/           # Existing: current default (unchanged)
└── octopus/                 # NEW
    ├── mod.rs               # Module index + re-exports
    ├── types.rs             # OctopusConfig, OctopusLayer, OctopusCodebook, OctopusTriplet
    ├── octahedral.rs        # oct_encode(n: [f32;3]) -> [f32;2], oct_decode([f32;2]) -> [f32;3]
    ├── triplet.rs           # decompose(u: &[f32]) -> Vec<Triplet>, recompose(triplets) -> Vec<f32>
    ├── codebook.rs          # build_norm_codebook(d, bits), build_oct_codebook(bits)
    ├── rotation.rs          # Re-export from turboquant (same WHT rotation)
    ├── encode.rs            # encode_triplet(t, codebooks) -> TripletIndices, joint_3x3_round
    ├── kv_cache.rs          # OctopusKVCache struct + QuantizedKVCache impl
    └── forward.rs           # forward_octopus(), score_path_decode()
```

### Key Types

```rust
// octopus/types.rs

/// Configuration for OCTOPUS KV cache
pub struct OctopusConfig {
    pub key_bits: u8,         // nominal bits per coordinate (actual: (b+1, b-1) split)
    pub val_bits: u8,
    pub seed: u64,            // deterministic rotation seed
    pub n_layers: usize,
    pub kv_dim: usize,        // must be power of 2 (WHT requirement)
    pub max_seq_len: usize,
    pub use_qjl_residual: bool,
    pub use_joint_rounding: bool,  // default: true (3×3 search)
}

/// Per-layer OCTOPUS state
pub struct OctopusLayer {
    pub rotation: Vec<f32>,        // sign-flipped WHT (reuse turboquant)
    pub key_codebook: OctopusCodebook,
    pub val_codebook: OctopusCodebook,
    pub qjl_signs: Option<Vec<f32>>,  // second rotation for QJL residual
}

/// Codebook pair for one side (K or V)
pub struct OctopusCodebook {
    pub norm_centroids: Vec<f32>,   // C_ρ: 2^(b-1) centroids on [0,1]
    pub norm_boundaries: Vec<f32>,  // midpoints for searchsorted
    pub oct_centroids: Vec<f32>,    // C_ξ: 2^(b+1) centroids on [-1,1]
    pub oct_boundaries: Vec<f32>,
    pub dir_bits: u8,               // b+1
    pub nrm_bits: u8,               // b-1
}

/// Packed indices for one triplet
pub struct TripletIndices {
    pub i_xi: u16,   // oct-coordinate ξ index
    pub i_eta: u16,  // oct-coordinate η index
    pub i_rho: u16,  // norm index
}
```

### Octahedral Map Core (octahedral.rs)

```rust
/// Encode unit vector on S² to [-1,1]² via octahedral map
/// Returns (ξ, η) coordinates
pub fn oct_encode(x: f32, y: f32, z: f32) -> (f32, f32) {
    let l = x.abs() + y.abs() + z.abs();
    let px = x / l;
    let py = y / l;
    let pz = z / l;
    if pz >= 0.0 {
        (px, py)
    } else {
        (px.signum() * (1.0 - py.abs()), py.signum() * (1.0 - px.abs()))
    }
}

/// Decode (ξ, η) from [-1,1]² back to unit vector on S²
pub fn oct_decode(xi: f32, eta: f32) -> (f32, f32, f32) {
    let r = 1.0 - xi.abs() - eta.abs();
    let (xp, yp) = if r >= 0.0 {
        (xi, eta)
    } else {
        (xi.signum() * (1.0 - eta.abs()), eta.signum() * (1.0 - xi.abs()))
    };
    let zp = r.max(0.0).copysign(1.0 - (r < 0.0) as i32 as f32); // handle sign
    let norm = (xp * xp + yp * yp + zp * zp).sqrt();
    (xp / norm, yp / norm, zp / norm)
}
```

### Bit Allocation Logic

```rust
// For nominal bit width b, the MSE-optimal split is:
//   dir_bits = b + 1  (for each oct-coordinate ξ, η)
//   nrm_bits = b - 1  (for triplet norm ρ)
// Total per triplet: 2*(b+1) + (b-1) = 3b + 1 bits
// vs. uniform: 3b bits
//
// At d=128, this gives 31-41% MSE reduction (verified empirically)
// The split is independent of d and total budget (key finding from paper)
```

### Integration with Existing Stack

```rust
// src/lib.rs
#[cfg(feature = "octopus")]
pub mod octopus;

// QuantizedKVCache trait is unchanged — OctopusKVCache implements it
// SP-KV composition works out of the box: SpKvQuantCache<OctopusKVCache>
// MaxSim scoring: forward_octopus() returns scores usable by MaxSim
```

### Feature Gate

```toml
# Cargo.toml
[features]
default = ["spectral_quant"]
full = ["turboquant", "spectral_quant", "octopus", "sp_kv", "maxsim"]
turboquant = []           # Legacy baseline
spectral_quant = []       # Current default (calibrated)
octopus = []              # NEW: data-oblivious triplet codec
sp_kv = []                # Composable with any quant backend
maxsim = []               # Late-interaction scoring
```

## Production Stack Positioning

```
GOAT Production Stack (after Bench 022):
  1. OCTOPUS        — default-on, data-oblivious, dominates SQ at all bit widths (Bench 022)
  2. SpectralQuant  — default-on, calibrated, useful for per-dimension water-fill adaptation
  3. TurboQuant     — legacy, kept for backward compatibility only

Decision flow:
  if need_extreme_compression(bits <= 3):
      use Octopus         # -22% to -49% MSE vs SQ, no calibration needed
  elif need_per_dimension_adaptation():
      use SpectralQuant   # water-fill adapts to eigenvalue spectrum
  else:
      use Octopus         # default choice, better quality at all bit widths
```

## GOAT Benchmark Plan

### T9: Synthetic MSE Sweep

File: `tests/goat_octopus_synthetic.rs`

```rust
// Sweep: d ∈ {64, 128, 256}, bits ∈ {2, 3, 4}, seeds = 64
// Metrics per (d, bits) combo:
//   - Reconstruction cosine similarity
//   - Per-coordinate MSE
//   - Inner-product absolute error
//   - Compression ratio
// Compare: TurboQuant vs OCTOPUS vs OCTOPUS-QJL
```

### T10: Compression Ratio Comparison

```rust
// At matched nominal bits, compare actual compression ratios:
//   TurboQuant: 2*b bits per triplet (uniform)
//   OCTOPUS:    2*(b+1) + (b-1) = 3b+1 bits per triplet
//   + norm storage (fp32, 4B per key)
// Report effective bits-per-scalar and KV× compression ratio
```

### T11: GOAT Proof Format

File: `.benchmarks/022_octopus_goat.md`

```markdown
# GOAT 022: OCTOPUS Octahedral KV Cache

## Configuration
- d ∈ {64, 128, 256}
- bits ∈ {2, 3, 4}
- 64 rotation seeds
- 8192 Gaussian keys, 64 Gaussian queries

## Results

### Reconstruction MSE (↓ better)
| d | bits | TurboQuant | OCTOPUS | Δ% |
|---|------|-----------|---------|-----|
| 128 | 2 | ... | ... | ... |
| 128 | 3 | ... | ... | ... |
| 128 | 4 | ... | ... | ... |

### Inner-Product Error (↓ better)
| d | bits | TurboQuant | OCTOPUS | OCTOPUS-QJL |
|---|------|-----------|---------|-------------|
| 128 | 2 | ... | ... | ... |

### Compression Ratio
| d | bits | TurboQuant KV× | OCTOPUS KV× |
|---|------|---------------|-------------|

## Verdict
[TO BE FILLED AFTER BENCHMARKS]
```

## Implementation Order

```
T1  octahedral.rs     — pure math, no dependencies, test immediately
T2  triplet.rs        — depends on octahedral for direction encoding
T3  codebook.rs       — depends on triplet for marginal sampling
T4  types.rs          — config + layer structs
T5  encode.rs         — depends on T1-T4, joint rounding
T6  kv_cache.rs       — depends on T4-T5, implements QuantizedKVCache
T7  forward.rs        — depends on T6, score-path decode
T8  feature gate      — wire up module, Cargo.toml
T9  GOAT synthetic    — benchmark TQ vs OCTOPUS
T10 GOAT compression  — ratio comparison
T11 GOAT proof        — record results
T12 README update     — document in production stack
```

## GOAT Results Summary (Bench 022)

### OCTOPUS vs SpectralQuant (default, calibrated) — d=128

| bits | SQ MSE   | OCT MSE  | MSE Δ%   | SQ Cos  | OCT Cos | Winner  |
|------|----------|----------|----------|---------|---------|---------|
| 2    | 0.1233   | 0.0962   | **-22%** | 0.9368  | 0.9512  | OCTOPUS |
| 3    | 0.0379   | 0.0263   | **-31%** | 0.9812  | 0.9870  | OCTOPUS |
| 4    | 0.0145   | 0.0074   | **-49%** | 0.9930  | 0.9963  | OCTOPUS |

OCTOPUS beats SpectralQuant at every bit width — zero calibration vs SQ's 256 samples.

### OCTOPUS vs TurboQuant (legacy) — d=128

| bits | TQ MSE   | OCT MSE  | MSE Δ%   | TQ Cos  | OCT Cos | Winner  |
|------|----------|----------|----------|---------|---------|---------|
| 2    | 0.1790   | 0.0962   | **-46%** | 0.9048  | 0.9512  | OCTOPUS |
| 3    | 0.0886   | 0.0263   | **-70%** | 0.9552  | 0.9870  | OCTOPUS |
| 4    | 0.0512   | 0.0074   | **-86%** | 0.9760  | 0.9963  | OCTOPUS |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| (b+1, b-1) split sub-optimal at d=64 | Medium | Sweep δ ∈ {-2..+2} at d=64 in T9 |
| Encode overhead too high for real-time | Low | Profile, optional `use_joint_rounding: false` fallback |
| d not divisible by 3 | Low | Zero-pad last triplet (paper does this) |
| Norm fp32 storage dominates at small d | Low | Accept as per-paper design (0.25 bpc at d=128) |

## Acceptance Criteria

- [x] `OctopusKVCache` implements `QuantizedKVCache` trait
- [x] All unit tests pass for octahedral encode/decode roundtrip
- [x] GOAT synthetic benchmark shows MSE improvement over SpectralQuant at d=128 (-22% to -49% MSE)
- [x] Feature gate `octopus` works independently (`cargo test --features octopus`)
- [x] `SpKvQuantCache<OctopusKVCache>` compiles (composition proof — 2 tests pass)
- [x] `.benchmarks/022_octopus_goat.md` populated with results
- [x] README updated with OCTOPUS section
- [x] OCTOPUS added to default features (GOAT proved: dominates SQ at all bit widths)