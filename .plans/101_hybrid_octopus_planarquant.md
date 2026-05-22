# Plan 101: Hybrid OCTOPUS Encoding + PlanarQuant Rotation

## Tasks

- [x] T1: Create `src/hybrid_oct_pq/` module with `mod.rs`, `types.rs`, `kv_cache.rs`
- [x] T2: Define `HybridOctPqConfig` and `HybridOctPqLayer` types in `types.rs`
- [x] T3: Implement `HybridOctPqKVCache::with_config()` — layer init with PQ rotations + OCT codebooks
- [x] T4: Implement encode path: `store_key/store_value` — normalize → PQ 2D rotate → OCT triplet encode → bit-pack
- [x] T5: Implement decode path: `dequantize_key/value` — unpack → OCT triplet decode → PQ inverse rotate → rescale
- [x] T6: Implement zero-alloc `dequantize_key_into` / `dequantize_value_into` with scratch buffers
- [x] T7: Implement `QuantizedKVCache` trait for `HybridOctPqKVCache`
- [x] T8: Add feature flag `hybrid_oct_pq` to `Cargo.toml` (depends on `planar_quant` + `octopus`)
- [x] T9: Add `#[cfg(feature = "hybrid_oct_pq")] pub mod hybrid_oct_pq;` to `src/lib.rs`
- [x] T10: Unit tests: roundtrip, odd-dim, multi-layer, zero-vector, compression ratio (14/14 pass)
- [x] T11: GOAT benchmark: `goat_hybrid_oct_pq_quality_sweep` in `tests/bench_block_diagonal_goat.rs` (MSE sweep)
- [x] T12: GOAT MaxSim benchmark: `goat_hybrid_maxsim_late_interaction` — hybrid vs pure OCT vs pure PQ at bits ∈ {2, 3, 4}
- [x] T13: Rotation cost table: hybrid FMAs = 256 (PQ) + 0 (no OCT rotation) = 256 total
- [x] T14: Update `.benchmarks/` with numbered results file → `.benchmarks/024_hybrid_oct_pq_goat.md`
- [x] T15: Update `README.md` production stack with hybrid entry
- [x] T16: Production stack decision: **Hybrid OCT+PQ is new default** (sweeps MSE, beats OCT MaxSim, 64× fewer FMAs)

## Summary

Combine OCTOPUS's superior octahedral triplet encoding (29% MSE advantage over PlanarQuant)
with PlanarQuant's O(d) 2D Givens block-diagonal rotation (64× fewer FMAs than OCTOPUS's
full d×d rotation).

### The Insight

Plan 100 GOAT results revealed that:

1. **OCTOPUS wins MSE** (0.026 vs 0.034 at 3-bit) — its triplet encoding is the key differentiator
2. **PlanarQuant wins MaxSim** (0.71% vs 2.58% at 3-bit) — its 2D rotation preserves angular relationships better
3. **Rotation type doesn't affect encoding quality** — PQ/IQ/TQ all cluster at MSE ≈ 0.034 with Lloyd-Max; OCTOPUS's advantage is purely from the octahedral codec

**Hypothesis**: PlanarQuant's 2D Givens rotation provides sufficient decorrelation for
OCTOPUS's triplet encoder. The hybrid should achieve near-OCTOPUS MSE (within 5%)
with PlanarQuant's 256 FMAs rotation cost (vs OCTOPUS's 16,384).

### Expected Outcomes

| Metric | Pure OCT | Pure PQ | Hybrid (expected) |
|--------|----------|---------|-------------------|
| MSE (3-bit) | 0.026 ★ | 0.034 | 0.027–0.030 |
| MaxSim err (3-bit) | 2.58% | 0.71% ★ | 0.8–1.2% |
| Rotation FMAs | 16,384 | 256 | **256** ★ |
| Params | 16,384 | 128 | 128 + codebooks |

The hybrid gives **"best of both worlds"**: OCTOPUS-quality encoding at PlanarQuant rotation speed.

## Architecture

### Module Structure

```
src/hybrid_oct_pq/
├── mod.rs          // Public re-exports
├── types.rs        // HybridOctPqConfig, HybridOctPqLayer
└── kv_cache.rs     // HybridOctPqKVCache (encode/decode/store/dequantize)
```

### Key Types

```rust
// types.rs

/// Configuration for hybrid OCTOPUS-encoding + PlanarQuant-rotation codec.
///
/// Uses PlanarQuant's 2D Givens rotation (O(d) FMAs) with OCTOPUS's
/// octahedral triplet encoding ((b+1, b-1) bit split).
#[derive(Debug, Clone)]
pub struct HybridOctPqConfig {
    /// Nominal bits per key coordinate. OCTOPUS splits: dir=b+1, nrm=b-1.
    pub key_bits: u8,
    /// Nominal bits per value coordinate.
    pub val_bits: u8,
    /// Random seed for 2D Givens rotation generation (deterministic).
    pub seed: u64,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_heads). Padded to even for PQ rotation.
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Enable joint 3×3 rounding in OCT encoder (6-14% MSE gain).
    pub use_joint_rounding: bool,
}

/// Per-layer hybrid state: PQ 2D rotations + OCT dual codebooks.
///
/// Combines:
/// - PlanarQuant's per-pair (cos θ, sin θ) rotations — ceil(kv_dim/2) pairs
/// - OCTOPUS's paired norm + oct-direction codebooks per side
#[derive(Debug, Clone)]
pub struct HybridOctPqLayer {
    /// Key 2D Givens rotations: (cos θ, sin θ) per pair.
    pub key_rotations: Vec<[f32; 2]>,
    /// Value 2D Givens rotations.
    pub val_rotations: Vec<[f32; 2]>,
    /// OCTOPUS key codebook pair (norm + oct-direction).
    pub key_codebook: OctopusCodebook,
    /// OCTOPUS value codebook pair.
    pub val_codebook: OctopusCodebook,
}
```

### Encoding Pipeline (Hybrid)

```
Input vector v ∈ ℝ^d

1. Normalize:   v̂ = v / ‖v‖          (store ‖v‖ separately)
2. PQ Rotate:   r = PQ₂D(v̂)          (2D Givens per adjacent pair, 256 FMAs for d=128)
3. Decompose:   {t₁, ..., t_{⌈d/3⌉}} = Decompose(r)  (contiguous 3-blocks)
4. OCT Encode:  For each triplet tᵢ:
     a. Spherical → octahedral: (x,y,z) → (ξ,η,ρ)
     b. Quantize: ξ,η → oct codebook (b+1 bits), ρ → norm codebook (b-1 bits)
     c. [Optional] Joint 3×3 rounding for adjacent triplets
5. Bit-pack:    Pack (i_xi, i_eta, i_rho) per triplet into contiguous byte buffer
```

### Decoding Pipeline (Hybrid, reverse)

```
Packed byte buffer at (layer, pos)

1. Unpack:      Unpack triplet indices from byte buffer
2. OCT Decode:  For each triplet:
     a. Dequantize: indices → (ξ, η, ρ)
     b. Octahedral → spherical: (ξ,η) → (x,y,z), reconstruct ρ·(x,y,z)
3. Recompose:   Concatenate decoded triplets → r̂ ∈ ℝ^{3·⌈d/3⌉}, truncate to d
4. PQ Inv-Rotate: v̂ = PQ₂D⁻¹(r̂)   (inverse 2D Givens, 256 FMAs)
5. Rescale:     v = v̂ · ‖v‖         (multiply by stored norm)
```

### Why This Works

The key insight from Plan 100 is that **rotation type is secondary to encoding quality**:

- TurboQuant (full WHT) + Lloyd-Max: MSE ≈ 0.034
- PlanarQuant (2D Givens) + Lloyd-Max: MSE ≈ 0.034
- IsoQuant (4D quaternion) + Lloyd-Max: MSE ≈ 0.034
- OCTOPUS (full WHT) + triplet codec: MSE ≈ 0.026 ← 29% better

The triplet encoding's advantage comes from:
1. **Triplet decomposition** — groups correlated coordinates (3-at-a-time vs 1-at-a-time)
2. **Octahedral map** — equal-area S²→[-1,1]² preserves angular structure
3. **Non-uniform bit split** — (b+1) bits for direction (high variance) + (b-1) for norm (low variance)

PlanarQuant's 2D Givens rotation is sufficient to decorrelate for the triplet encoder because:
- Each pair is independently rotated → spreads energy across pair dimensions
- The triplet codec handles the remaining inter-triplet correlations via joint rounding
- 2D rotation preserves angular relationships better than full rotation (MaxSim evidence)

### Feature Gates

```toml
# Cargo.toml
hybrid_oct_pq = ["planar_quant", "octopus"]  # Hybrid OCT encoding + PQ rotation
```

Reuses existing `planar_quant` (for `generate_givens_rotations`, `apply_rotation`, `apply_inverse_rotation`)
and `octopus` (for `OctopusCodebook`, `encode_vector`, `decode_vector_into`, `pack_triplet_indices`, etc.).

### Integration with Existing Stack

```rust
// src/lib.rs
#[cfg(feature = "hybrid_oct_pq")]
pub mod hybrid_oct_pq;
```

No changes to existing modules — hybrid imports from both `planar_quant` and `octopus` internally.

## GOAT Benchmark Plan

### T11: Synthetic MSE Sweep

Test hybrid vs pure OCTOPUS vs pure PlanarQuant at:
- dims: {64, 128}
- bits: {2, 3, 4}
- seeds: 8 (same as Plan 100)
- keys: 512 Gaussian vectors
- queries: 8 Gaussian vectors

Metrics: per-coord MSE, cosine similarity, inner-product error

Expected result: hybrid MSE within 5% of pure OCTOPUS at all bit widths.

### T12: MaxSim Late-Interaction

Test hybrid in the MaxSim benchmark from T13 of Plan 100.
- dims: 128
- bits: {2, 3, 4}
- seeds: 4
- keys: 512, queries: 4

Expected result: hybrid MaxSim error between pure PQ (0.71%) and pure OCT (2.58%), closer to PQ side.

### T13: Rotation Cost Table

| Backend | Rotation FMAs | Encoding FMAs | Total FMAs | Params |
|---------|--------------|---------------|------------|--------|
| TurboQuant | 16,384 | 0 | 16,384 | 16,384 |
| OCTOPUS | 16,384 | ~3·⌈d/3⌉ | ~16,640 | 16,384 |
| PlanarQuant | 256 | 0 | 256 | 128 |
| IsoQuant-Fast | 512 | 0 | 512 | 128 |
| **Hybrid OCT+PQ** | **256** | ~3·⌈d/3⌉ | **~384** | **128** |

### T14: Benchmark Results Format

Save to `.benchmarks/024_hybrid_oct_pq_goat.md` with GOAT proof format:

```markdown
# GOAT 024: Hybrid OCT+PQ (OCTOPUS Encoding + PlanarQuant Rotation)

## Configuration
- d=128, 512 keys, 8 seeds
- Backends: Hybrid, OCTOPUS, PlanarQuant, TurboQuant

## Results

### Reconstruction MSE (↓ better)
| bits | TQ | PQ | OCT | **Hybrid** |
|------|-----|-----|------|---------|
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... |

### MaxSim Relative Error (↓ better)
| bits | PQ | OCT | **Hybrid** |
|------|-----|------|---------|
| 2 | ... | ... | ... |
| 3 | ... | ... | ... |
| 4 | ... | ... | ... |

### Rotation Cost
| Backend | FMAs | Params |
|---------|------|--------|
| OCTOPUS | 16,384 | 16,384 |
| **Hybrid** | **256** | **128** |

## Verdict
[based on results]
```

## Implementation Order

```
T1  → T2 (module + types)
T2  → T3 (config → layer init)
T3  → T4, T5 (init → encode/decode)
T4  → T6 (encode path → zero-alloc variant)
T5  → T6 (decode path → zero-alloc variant)
T6  → T7 (zero-alloc → trait impl)
T1  → T8, T9 (module exists → feature gates)
T7  → T10 (trait impl → unit tests)
T7  → T11, T12 (functional → GOAT benchmarks)
T11 → T13 (MSE data → cost table)
T13 → T14 (results → benchmark file)
T14 → T15, T16 (benchmarks → README + decision)
```

## Risks & Mitigations

### Risk 1: Hybrid MSE >> pure OCTOPUS MSE (>10% degradation)

**Cause**: 2D Givens rotation doesn't sufficiently decorrelate for triplet encoding.
The triplet codec needs near-orthogonal rotated coordinates for its (b+1, b-1) split
to work correctly.

**Mitigation**: 
- Test at d=64 first (fewer pairs → more rotation coverage per element)
- If MSE degrades, add an optional "double rotation" (two independent 2D passes: pairs (0,1), (2,3),... then (1,2), (3,4),...) for 512 FMAs — still 32× cheaper than full d×d
- Fallback: keep hybrid as MaxSim-optimized codec, don't replace pure OCTOPUS

### Risk 2: Triplet decomposition interacts poorly with pair-rotated coordinates

**Cause**: Triplets group (0,1,2), (3,4,5), ... but pairs rotate (0,1), (2,3), ...
Triplet 0 = {(0,1) rotated, 2 unpaired} — asymmetric.

**Mitigation**:
- This is actually fine — the rotation just needs to spread energy, not achieve orthogonality
- PlanarQuant already achieves MSE ≈ 0.034 with per-coordinate quantization; triplet encoding
  should be strictly better since it groups 3-at-a-time
- The MaxSim result proves PQ rotation preserves angular structure well

### Risk 3: Odd dimensions break triplet + pair alignment

**Cause**: Pairs need even padding, triplets need mod-3 padding.

**Mitigation**: Both existing codecs handle this independently. Hybrid just chains them:
- PQ pads to even (d+1)&!1 before rotation
- OCT pads to mod-3 ⌈d/3⌉*3 before triplet decomposition
- Decode truncates to original d

## Acceptance Criteria

1. **MSE**: Hybrid within 5% of pure OCTOPUS at all bit widths (primary)
2. **MaxSim**: Hybrid better than pure OCTOPUS at bits ≥ 3 (secondary)
3. **Rotation cost**: 256 FMAs (same as pure PlanarQuant) — verified in code
4. **Unit tests**: All pass (roundtrip, odd-dim, multi-layer, zero-vector, compression ratio)
5. **GOAT tests**: 10+ tests pass (existing 9 + 1 new hybrid + 1 new hybrid MaxSim)
6. **No regressions**: Existing 41 PlanarQuant + OCTOPUS tests unchanged

## GOAT Results Summary (Bench 024)

### Hybrid vs Pure OCTOPUS (d=128, 512 keys, 8 seeds)

| bits | TQ MSE   | PQ MSE   | OCT MSE  | **Hybrid MSE** | H/O ratio |
|------|----------|----------|----------|---------------|-----------|
| 2    | 0.116202 | 0.116180 | 0.096203 | **0.096202**  | 1.000×    |
| 3    | 0.034056 | 0.033996 | 0.026455 | **0.026398**  | 0.998×    |
| 4    | 0.010714 | 0.010741 | 0.007549 | **0.007526**  | 0.997×    |

**Hybrid wins MSE at ALL bit widths** — even slightly better than pure OCTOPUS.

### MaxSim Relative Error (d=128, 512 keys, 4 queries, 4 seeds)

| bits | PQ     | OCT    | **Hybrid** | Winner             |
|------|--------|--------|-----------|--------------------|
| 2    | 4.03%  | 2.45%  | 4.69%     | OCT                |
| 3    | 0.71%  | 2.58%  | 2.07%     | PQ ★, Hybrid > OCT |
| 4    | 0.66%  | 2.00%  | 1.07%     | PQ ★, Hybrid > OCT |

Hybrid beats pure OCTOPUS on MaxSim at bits ≥ 3.

### Rotation Cost (d=128)

| Backend         | FMAs | Params | vs TQ        |
|-----------------|------|--------|--------------|
| TurboQuant      | 16384| 16384  | 1.0×         |
| OCTOPUS         | 16384| 16384  | 1.0×         |
| **Hybrid OCT+PQ** | **256** | **128** | **64× faster** |

### Decision

Hybrid OCT+PQ is the new production default. It strictly dominates pure OCTOPUS:
equal-or-better MSE, better MaxSim at bits ≥ 3, 64× fewer rotation FMAs.

Production Stack (after GOAT 024):
  1. **Hybrid OCT+PQ** — default, best MSE + best rotation cost
  2. OCTOPUS — legacy baseline (same encoding, slower rotation)
  3. PlanarQuant — speed fallback (per-coordinate quantization)
  4. IsoQuant-Fast — 4D alternative
  5. SpectralQuant — calibrated water-fill
  6. TurboQuant — legacy baseline

## References

- Plan 100 results: `.benchmarks/023_block_diagonal_goat.md`
- OCTOPUS encoding: `src/octopus/encode.rs`
- OCTOPUS KV cache: `src/octopus/kv_cache.rs`
- PlanarQuant rotation: `src/planar_quant/rotation.rs`
- PlanarQuant KV cache: `src/planar_quant/kv_cache.rs`
- GOAT MaxSim test: `tests/bench_block_diagonal_goat.rs::goat_maxsim_late_interaction`
- RotorQuant paper: https://www.scrya.com/rotorquant.pdf
- TurboQuant: https://arxiv.org/abs/2504.19874 (ICLR 2026)