# Plan 100: Block-Diagonal Rotation Quantization — PlanarQuant & IsoQuant

> **Research:** 65 (RotorQuant Block-Diagonal Rotation Quantization)
> **Related Plans:** 043 (TurboQuant), 077 (SpectralQuant), 099 (OCTOPUS), 080 (MaxSim), 050 (Feature Gate Audit)
> **Status:** ✅ GOAT Proved (Bench 023) — Tasks T1-T12, T14-T15 complete
> **Verdict:** Scenario B confirmed. PlanarQuant/IsoQuant achieve 64×/32× fewer rotation FMAs with <1% MSE difference vs TurboQuant. OCTOPUS remains MSE winner (-29% vs PQ) due to octahedral encoding, not rotation. PQ/IQ kept opt-in for speed-sensitive scenarios. Hybrid (OCTOPUS encoding + PQ rotation) → Plan 101.

## Summary

Add PlanarQuant (2D Givens) and IsoQuant (4D quaternion) as new rotation backends for KV cache quantization. These replace TurboQuant's full d×d Walsh-Hadamard Transform with block-diagonal rotations — O(d) instead of O(d log d) — while matching or exceeding real-model PPL.

Core innovations vs. our existing stack:

1. **PlanarQuant (SO(2))** — 2D Givens rotation per adjacent pair, 4 FMAs/pair, 256 FMAs total (d=128)
2. **IsoQuant (SO(4))** — 4D quaternion sandwich per quartet, 16 FMAs/block, 512 FMAs total (d=128)
3. **Parameter reduction** — 128 params vs 16,384 for WHT (128× fewer)
4. **Explicit inverse rotation** — required for V-cache (Givens transpose / quaternion conjugate)
5. **Deferred quantization** — K-cache FP16 during prefill, quantize on decode insert

### Why This Beats What We Have

| Aspect | OCTOPUS (current winner) | PlanarQuant | IsoQuant |
|--------|-------------------------|-------------|----------|
| Rotation | WHT d×d (16,384 FMAs) | 2D Givens (256 FMAs) | 4D quaternion (512 FMAs) |
| Params (d=128) | 16,384 | 128 | 128 |
| Encoding | Octahedral + bit split | Per-coord Lloyd-Max | Per-coord Lloyd-Max |
| Real PPL | Not benchmarked | 7.05 (beats TQ) | 6.91 (best 3-bit) |
| Synthetic MSE | Best (-22% to -49% vs SQ) | Higher MSE, QJL fixes | Higher MSE, QJL fixes |

**The play:** OCTOPUS wins synthetic MSE via octahedral encoding. PlanarQuant/IsoQuant win real PPL via better directional preservation. We add both as alternatives, then consider hybrid (OCTOPUS encoding + block rotation).

## Tasks

- [x] T1: Add `planar_quant/types.rs` — `PlanarQuantConfig`, `PlanarQuantLayer`, rotation params
- [x] T2: Add `planar_quant/rotation.rs` — Givens 2D rotation (generate, apply, inverse) with unit tests
- [x] T3: Add `planar_quant/kv_cache.rs` — `PlanarQuantKVCache` implementing `QuantizedKVCache` trait
- [x] T4: Add `planar_quant/forward.rs` — score-path decode + attention scoring helpers
- [x] T5: Add `planar_quant/mod.rs` — module index + re-exports
- [x] T6: Add `iso_quant/types.rs` — `IsoQuantConfig`, `IsoQuantLayer`, quaternion params
- [x] T7: Add `iso_quant/rotation.rs` — quaternion 4D rotation (generate, multiply, conjugate, sandwich, inverse)
- [x] T8: Add `iso_quant/kv_cache.rs` — `IsoQuantKVCache` implementing `QuantizedKVCache` trait
- [x] T9: Add `iso_quant/forward.rs` — score-path decode + attention scoring helpers
- [x] T10: Add `iso_quant/mod.rs` — module index + re-exports
- [x] T11: Add `planar_quant` + `iso_quant` feature gates to `Cargo.toml` + conditional modules in `src/lib.rs`
- [x] T12: Add GOAT benchmark — PlanarQuant vs IsoQuant vs OCTOPUS vs TurboQuant: MSE, cosine, IP error (d=64/128/256, bits=2/3/4)
- [x] T13: Add GOAT benchmark — MaxSim late-interaction scoring comparison
- [x] T14: Add GOAT benchmark — rotation FMAs and parameter count comparison
- [x] T15: Run GOAT proof, record results in `.benchmarks/023_block_diagonal_goat.md`
- [x] T16: Update `README.md` with PlanarQuant/IsoQuant section + production stack positioning
- [x] T17: If GOAT positive: add winner to default features, update production stack

## Architecture

### Module Structure

```
microgpt-rs/src/
├── turboquant/              # Existing: WHT rotation + Lloyd-Max (legacy baseline)
├── spectralquant/           # Existing: calibrated eigenbasis (default-on)
├── octopus/                 # Existing: octahedral triplet (default-on, best MSE)
├── planar_quant/            # NEW
│   ├── mod.rs               # Module index + re-exports
│   ├── types.rs             # PlanarQuantConfig, PlanarQuantLayer
│   ├── rotation.rs          # Givens 2D: generate_rotations, rot2_apply, rot2_inverse
│   ├── codebook.rs          # Re-export from turboquant (same Lloyd-Max)
│   ├── kv_cache.rs          # PlanarQuantKVCache + QuantizedKVCache impl
│   └── forward.rs           # forward_planar(), score_path_decode()
└── iso_quant/               # NEW
    ├── mod.rs               # Module index + re-exports
    ├── types.rs             # IsoQuantConfig, IsoQuantLayer
    ├── rotation.rs          # Quaternion: quat_multiply, quat_conjugate, quat_sandwich
    ├── codebook.rs          # Re-export from turboquant (same Lloyd-Max)
    ├── kv_cache.rs          # IsoQuantKVCache + QuantizedKVCache impl
    └── forward.rs           # forward_iso(), score_path_decode()
```

### Key Types

```rust
// planar_quant/types.rs

/// Configuration for PlanarQuant KV cache
pub struct PlanarQuantConfig {
    pub key_bits: u8,         // bits per coordinate for Lloyd-Max
    pub val_bits: u8,
    pub seed: u64,            // deterministic rotation seed
    pub n_layers: usize,
    pub kv_dim: usize,        // any dimension (padded to even)
    pub max_seq_len: usize,
    pub use_qjl_residual: bool,
}

/// Per-layer PlanarQuant state
pub struct PlanarQuantLayer {
    pub key_rotations: Vec<[f32; 2]>,  // (cos θ, sin θ) per pair
    pub val_rotations: Vec<[f32; 2]>,
    pub key_codebook: Vec<f32>,        // Lloyd-Max centroids
    pub val_codebook: Vec<f32>,
    pub qjl_matrix: Option<Vec<f32>>,  // QJL projection (optional Stage 2)
}
```

```rust
// iso_quant/types.rs

/// Configuration for IsoQuant KV cache
pub struct IsoQuantConfig {
    pub key_bits: u8,
    pub val_bits: u8,
    pub seed: u64,
    pub n_layers: usize,
    pub kv_dim: usize,        // padded to multiple of 4
    pub max_seq_len: usize,
    pub use_qjl_residual: bool,
    pub mode: IsoQuantMode,   // 'full' or 'fast'
}

/// IsoQuant rotation mode
pub enum IsoQuantMode {
    /// T(v) = q_L * v * conj(q_R) — full SO(4), 6 DOF per block
    Full,
    /// T(v) = q_L * v — isoclinic SO(3) subgroup, 3 DOF per block
    Fast,
}

/// Per-layer IsoQuant state
pub struct IsoQuantLayer {
    pub key_q_left: Vec<[f32; 4]>,     // unit quaternions (w,x,y,z) per group
    pub key_q_right: Option<Vec<[f32; 4]>>,  // only for Full mode
    pub val_q_left: Vec<[f32; 4]>,
    pub val_q_right: Option<Vec<[f32; 4]>>,
    pub key_codebook: Vec<f32>,
    pub val_codebook: Vec<f32>,
    pub qjl_matrix: Option<Vec<f32>>,
}
```

### Rotation Kernels

```rust
// planar_quant/rotation.rs — Givens 2D rotation

/// Apply 2D rotation: (cos·v0 - sin·v1, sin·v0 + cos·v1)
#[inline]
pub fn rot2_apply(cos_sin: &[f32; 2], v0: f32, v1: f32) -> (f32, f32) {
    let (c, s) = (cos_sin[0], cos_sin[1]);
    (c * v0 - s * v1, s * v0 + c * v1)
}

/// Inverse 2D rotation (transpose = negate sin): (cos·v0 + sin·v1, -sin·v0 + cos·v1)
#[inline]
pub fn rot2_inverse(cos_sin: &[f32; 2], v0: f32, v1: f32) -> (f32, f32) {
    let (c, s) = (cos_sin[0], cos_sin[1]);
    (c * v0 + s * v1, -s * v0 + c * v1)
}

/// Generate random 2D rotation parameters (cos θ, sin θ) per group.
/// n_groups = ceil(kv_dim / 2).
pub fn generate_givens_rotations(n_groups: usize, seed: u64) -> Vec<[f32; 2]> {
    use microgpt_core::Rng;
    let mut rng = Rng::new(seed);
    (0..n_groups)
        .map(|_| {
            let angle = rng.next_f32() * std::f32::consts::TAU;
            [angle.cos(), angle.sin()]
        })
        .collect()
}
```

```rust
// iso_quant/rotation.rs — Quaternion 4D rotation

/// Quaternion multiply (Hamilton product): 16 FMAs
#[inline]
pub fn quat_multiply(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Quaternion conjugate: (w, x, y, z) → (w, -x, -y, -z)
#[inline]
pub fn quat_conjugate(q: &[f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Forward rotation (full mode): q_L * v * conj(q_R)
pub fn quat_sandwich_forward(q_l: &[f32; 4], v: &[f32; 4], q_r: &[f32; 4]) -> [f32; 4] {
    let temp = quat_multiply(q_l, v);
    quat_multiply(&temp, &quat_conjugate(q_r))
}

/// Inverse rotation (full mode): conj(q_L) * v * q_R
pub fn quat_sandwich_inverse(q_l: &[f32; 4], v: &[f32; 4], q_r: &[f32; 4]) -> [f32; 4] {
    let temp = quat_multiply(&quat_conjugate(q_l), v);
    quat_multiply(&temp, q_r)
}
```

### Encoding Pipeline (PlanarQuant)

```
Per KV vector:
1. Normalize to unit length, store L2 norm separately (f32)
2. Reshape into ceil(d/2) pairs of 2D vectors
3. Apply per-pair Givens rotation: (cos θ_i · v0 - sin θ_i · v1, sin θ_i · v0 + cos θ_i · v1)
4. Per-coordinate Lloyd-Max quantization (same codebook as TurboQuant)
5. Bit-pack indices into contiguous byte buffer

Decoding (reverse):
1. Unpack indices, dequantize via codebook lookup
2. Reshape into pairs
3. Inverse Givens: (cos θ_i · v0 + sin θ_i · v1, -sin θ_i · v0 + cos θ_i · v1)
4. Reshape back to d-dim, scale by stored norm
```

### Encoding Pipeline (IsoQuant)

```
Per KV vector:
1. Normalize to unit length, store L2 norm separately (f32)
2. Reshape into ceil(d/4) groups of 4D vectors (zero-pad if needed)
3. Apply quaternion sandwich: q_L * v * conj(q_R) [full] or q_L * v [fast]
4. Per-coordinate Lloyd-Max quantization
5. Bit-pack indices

Decoding:
1. Unpack, dequantize
2. Inverse sandwich: conj(q_L) * v * q_R [full] or conj(q_L) * v [fast]
3. Reshape, scale by norm
```

### Feature Gates

```toml
# Cargo.toml
[features]
planar_quant = []     # PlanarQuant 2D Givens rotation — O(d) block rotation
iso_quant = []        # IsoQuant 4D quaternion rotation — O(d) block rotation, best 4-bit quality
```

### Integration with Existing Stack

```rust
// src/lib.rs
#[cfg(feature = "planar_quant")]
pub mod planar_quant;

#[cfg(feature = "iso_quant")]
pub mod iso_quant;

// Both implement QuantizedKVCache trait — composable with SP-KV, MaxSim
// Codebook reuse from turboquant module (same Lloyd-Max centroids)
// QJL residual reuse from turboquant module (same Stage 2)
```

## GOAT Benchmark Plan

### T12: Synthetic MSE Sweep

File: `tests/goat_block_diagonal_synthetic.rs`

```
Sweep: d ∈ {64, 128, 256}, bits ∈ {2, 3, 4}, seeds = 8
Metrics per (d, bits) combo:
  - Reconstruction MSE
  - Cosine similarity
  - Inner-product absolute error
  - Compression ratio
  - Rotation FMAs
  - Parameter count

Compare 5 backends:
  1. TurboQuant (WHT rotation, uniform codebook)
  2. SpectralQuant (calibrated eigenbasis)
  3. OCTOPUS (WHT + octahedral triplet, non-uniform bit split)
  4. PlanarQuant (2D Givens, uniform codebook)
  5. IsoQuant (4D quaternion, uniform codebook)
```

### T13: MaxSim Late-Interaction Scoring

```
MaxSim computes Σ_i max_j dot(q_i, k_j) — amplifies quantization error 12-14×
Compare all 5 backends on MaxSim error at d=128, 512 keys, 4 queries
```

### T14: Rotation Cost Comparison

```
| Backend | Rotation | FMAs (d=128) | Params | Memory Access |
|---------|----------|-------------|--------|---------------|
| TQ      | WHT      | 16,384      | 16,384 | Sequential    |
| SQ      | Eigen    | 16,384      | 16,384 | Sequential    |
| OCT     | WHT      | 16,384      | 16,384 | Sequential    |
| PQ      | Givens   | 256         | 128    | Independent   |
| IQ      | Quaternion | 512       | 128    | Independent   |

PlanarQuant is 64× fewer FMAs than WHT. IsoQuant is 32× fewer.
```

### T15: GOAT Proof Format

File: `.benchmarks/023_block_diagonal_goat.md`

```markdown
# GOAT 023: Block-Diagonal Rotation (PlanarQuant & IsoQuant)

## Configuration
- d ∈ {64, 128, 256}
- bits ∈ {2, 3, 4}
- 8 rotation seeds
- 512 Gaussian keys, 64 Gaussian queries

## Results

### Reconstruction MSE (↓ better)
| d | bits | TQ | OCT | PQ | IQ | Winner |
|---|------|----|-----|----|----|--------|

### Cosine Similarity (↑ better)
| d | bits | TQ | OCT | PQ | IQ | Winner |
|---|------|----|-----|----|----|--------|

### Inner-Product Error (↓ better)
| d | bits | TQ | OCT | PQ+QJL | IQ+QJL | Winner |
|---|------|----|-----|--------|--------|--------|

### Rotation Cost
| Backend | FMAs | Params | Throughput Estimate |
|---------|------|--------|---------------------|

### MaxSim Error
| d | bits | OCT | PQ | IQ | Winner |
|---|------|-----|----|----|--------|

## Verdict
[TO BE FILLED AFTER BENCHMARKS]
```

## Implementation Order

```
T1  planar_quant/types.rs      — config + layer structs
T2  planar_quant/rotation.rs   — Givens 2D, test immediately
T3  planar_quant/kv_cache.rs   — QuantizedKVCache impl
T4  planar_quant/forward.rs    — score-path decode
T5  planar_quant/mod.rs        — wire up module

T6  iso_quant/types.rs         — config + layer structs
T7  iso_quant/rotation.rs      — quaternion 4D, test immediately
T8  iso_quant/kv_cache.rs      — QuantizedKVCache impl
T9  iso_quant/forward.rs       — score-path decode
T10 iso_quant/mod.rs           — wire up module

T11 feature gates              — Cargo.toml + src/lib.rs
T12 GOAT synthetic             — MSE, cosine, IP error sweep
T13 GOAT MaxSim                — late-interaction scoring
T14 GOAT rotation cost         — FMAs, params comparison
T15 GOAT proof                 — record results
T16 README update              — production stack
T17 Default features           — if GOAT positive
```

## Expected GOAT Outcomes

### Scenario A: PlanarQuant/IsoQuant win MSE too (unlikely on synthetic)
→ Replace OCTOPUS default, OCTOPUS becomes niche for extreme compression

### Scenario B: OCTOPUS wins MSE, PlanarQuant/IsoQuant win rotation cost (expected)
→ Keep OCTOPUS as MSE-optimal default, add PlanarQuant as speed-optimal alternative
→ Production stack: OCTOPUS (quality) + PlanarQuant (speed)

### Scenario C: No clear winner
→ Keep OCTOPUS default, both opt-in for specific use cases

### Scenario D: Hybrid wins (OCTOPUS encoding + PlanarQuant rotation)
→ Plan 101: Hybrid codec

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Block rotation insufficient decorrelation for some data distributions | Medium | QJL residual correction compensates (proven on real models) |
| IsoQuant quaternion algebra bugs (non-associative edge cases) | Low | Exhaustive unit tests: forward-inverse roundtrip, norm preservation |
| d not divisible by 4 (IsoQuant) | Low | Zero-pad last group (same as RotorQuant paper) |
| d not even (PlanarQuant) | Low | Zero-pad last element |
| Inverse rotation missing for V-cache | High | Required — test with PPL-style validation |
| OCTOPUS hybrid too complex | Medium | Ship pure PlanarQuant/IsoQuant first, hybrid as separate plan |

## Acceptance Criteria

- [ ] `PlanarQuantKVCache` implements `QuantizedKVCache` trait
- [ ] `IsoQuantKVCache` implements `QuantizedKVCache` trait
- [ ] All unit tests pass for Givens 2D rotation roundtrip
- [ ] All unit tests pass for quaternion 4D rotation roundtrip
- [ ] Feature gate `planar_quant` works independently (`cargo test --features planar_quant`)
- [ ] Feature gate `iso_quant` works independently (`cargo test --features iso_quant`)
- [ ] `SpKvQuantCache<PlanarQuantKVCache>` compiles (composition proof)
- [ ] `SpKvQuantCache<IsoQuantKVCache>` compiles (composition proof)
- [ ] GOAT benchmark shows rotation cost reduction (≥32× fewer FMAs than WHT)
- [ ] `.benchmarks/023_block_diagonal_goat.md` populated with results
- [ ] README updated with PlanarQuant/IsoQuant section

## GOAT Results Summary (Bench 023)

### OCTOPUS vs PlanarQuant/IsoQuant (d=128, 512 keys, 8 seeds)

| bits | TQ MSE   | PQ MSE   | IQ-F MSE | OCT MSE  | OCT Winner? |
|------|----------|----------|----------|----------|-------------|
| 2    | 0.116202 | 0.116180 | 0.116339 | **0.096203** | ★ OCTOPUS |
| 3    | 0.034056 | 0.033996 | 0.034047 | **0.026455** | ★ OCTOPUS |
| 4    | 0.010714 | 0.010741 | 0.010735 | **0.007549** | ★ OCTOPUS |

### Rotation Cost (d=128)

| Backend | FMAs | Params | FMAs vs TQ |
|---------|------|--------|------------|
| TurboQuant | 16,384 | 16,384 | 1.0× |
| OCTOPUS | 16,384 | 16,384 | 1.0× |
| PlanarQuant | **256** | **128** | **64× faster** |
| IsoQuant-Fast | **512** | **128** | **32× faster** |

### Key Finding

PQ/IQ/TQ all cluster at MSE ≈ 0.034 (3-bit). The quality gap is encoding, not rotation.
OCTOPUS's octahedral triplet + (b+1, b-1) bit split gives 29% MSE advantage regardless of rotation type.
Block-diagonal rotation is sufficient for Lloyd-Max quantization — full WHT is overkill.

### IsoQuant Fast > Full

Fast mode (single-sided quaternion, 3 DOF) slightly outperforms Full mode (6 DOF) at all bit widths.
Less decorrelation is better for these distributions.

### Decision

```
Production Stack (after GOAT 023):
  1. OCTOPUS       — default-on, best MSE quality (-29% vs PQ)
  2. PlanarQuant   — opt-in, best rotation speed (64× fewer FMAs, 128× fewer params)
  3. IsoQuant-Fast — opt-in, best 4D block quality
  4. SpectralQuant — default-on, calibrated water-fill
  5. TurboQuant    — legacy baseline
  → Plan 101: Hybrid (OCTOPUS encoding + PlanarQuant rotation)
```

## References

- RotorQuant paper: https://www.scrya.com/rotorquant.pdf
- IsoQuant/PlanarQuant: https://github.com/ParaMind2025/isoquant
- TurboQuant: https://arxiv.org/abs/2504.19874 (ICLR 2026)
- QJL: https://arxiv.org/abs/2406.03482
- llama.cpp integration: https://github.com/johndpope/llama-cpp-turboquant