# Research 65: RotorQuant / PlanarQuant / IsoQuant — Block-Diagonal Rotation for KV Cache Quantization

**Source:** [RotorQuant paper](https://www.scrya.com/rotorquant.pdf) (Pope, March 2026) + [local reference](/.raw/rotorquant/) + [IsoQuant/PlanarQuant](https://github.com/ParaMind2025/isoquant)
**Related:** Research 20 (TurboQuant), Research 39 (SpectralQuant), Research 63 (OCTOPUS)
**Date:** 2025-07-05

## TL;DR

RotorQuant replaces TurboQuant's d×d random orthogonal matrix with Clifford algebra rotors (3D blocks), achieving 44× fewer parameters and 10-31× faster GPU kernels while matching attention fidelity. The evolution continued with PlanarQuant (2D Givens, 4 FMAs/pair) and IsoQuant (4D quaternion, 16 FMAs/block), which both beat TurboQuant on real PPL benchmarks. The core insight: **block-diagonal rotations suffice for KV cache decorrelation** because real attention vectors live on low-rank manifolds.

## Key Ideas

### 1. Block-Diagonal Rotation Hierarchy

| Method | Algebra | Group Size | FMAs (d=128) | Params (d=128) | Status |
|--------|---------|-----------|--------------|----------------|--------|
| TurboQuant | WHT butterfly | 128 (full) | 16,384 | 16,384 | Baseline |
| RotorQuant | Cl(3,0) rotor | 3 | ~2,400 | 372 | Research |
| IsoQuant | Quaternion SO(4) | 4 | 512 | 128 | **Production** |
| PlanarQuant | Givens SO(2) | 2 | 256 | 128 | **Production** |

Each step trades algebraic richness for speed. Simpler rotations work *better* on real data.

### 2. Why Block-Diagonal Wins

- **Real KV vectors are not i.i.d. Gaussian** — they live on low-rank manifolds shaped by attention patterns
- Full d×d rotation over-decorrelates, scrambling directional structure
- Small block rotations preserve within-block structure while providing enough decorrelation for scalar quantization
- This explains why RotorQuant beats TurboQuant on top-1/top-5 retrieval at 4K context (Table 6)

### 3. Deferred Quantization (Critical for PPL)

K-cache stays FP16 during prefill → converts to quantized post-prefill. This eliminates error compounding during the attention-heavy prefill phase and gives 3× better PPL than round-trip quantization.

### 4. Real PPL Results (Llama 3.1 8B, WikiText-2, ctx=2048)

| Config (K/V) | PPL | vs FP16 (6.63) | Compression |
|---|---|---|---|
| iso3 / iso3 | 6.91 | +4.2% | 10.3× |
| planar3 / planar3 | 7.05 | +6.3% | 10.3× |
| turbo3 / turbo3 | 7.07 | +6.6% | 10.3× |
| planar3 / turbo3 | 6.68 | +0.8% | — |

**Both IsoQuant and PlanarQuant beat TurboQuant PPL at same compression ratio.**

### 5. Decode Speed (RTX 5090, Qwen2.5-3B, K-only)

| Cache K | Decode tok/s |
|---------|-------------|
| planar3 | **367** |
| FP16 | 356 |

PlanarQuant decode is *faster than FP16* due to reduced memory bandwidth.

### 6. Parameter Efficiency

| d | TurboQuant | RotorQuant | Ratio |
|---|-----------|-----------|-------|
| 128 | 16,399 | 372 | 44× |
| 256 | 65,540 | 358 | 183× |
| 512 | 262,148 | 698 | 376× |
| 4096 | 16,777,220 | 5,478 | 3,063× |

Parameter savings scale super-linearly with dimension.

### 7. Inverse Rotation Matters for V-Cache

V dequant must apply explicit inverse rotation (inverse Givens or inverse quaternion). TurboQuant's WHT doesn't need this due to self-canceling Hadamard properties. Missing this caused PPL 15,369 → fixed to 7.05.

## Distillation for Our Stack

### What We Can Use

1. **PlanarQuant (2D Givens)** — simplest, fastest, already beats TurboQuant on PPL
   - 4 FMAs per 2D pair (cos·v0 - sin·v1, sin·v0 + cos·v1)
   - 256 FMAs total for d=128 vs TurboQuant's 16,384
   - Pairs align to any even dimension (most common)
   - Trivially vectorizable — no quaternion algebra needed

2. **IsoQuant (4D quaternion)** — best quality at 4-bit
   - 16 FMAs per 4D block (Hamilton product)
   - 512 FMAs total for d=128
   - 4D blocks align to powers-of-2 head dims (128 = 32 groups)
   - Two modes: 'full' (q_L v q̄_R, 6 DOF) or 'fast' (q_L v, 3 DOF)

3. **QJL Residual Correction** — already implemented in our TurboQuant, reusable as-is

4. **Deferred Quantization** — K-cache stays FP16 during prefill, quantize on decode insert

### What We Already Have (OCTOPUS)

- OCTOPUS uses triplet decomposition (3D blocks) + octahedral map + (b+1, b-1) bit split
- OCTOPUS dominates SpectralQuant at all bit widths (our GOAT Bench 022)
- OCTOPUS is data-oblivious (no calibration), uses WHT rotation (same as TurboQuant)
- OCTOPUS's octahedral map provides geometric structure that PlanarQuant/IsoQuant don't

### Key Differences: OCTOPUS vs PlanarQuant/IsoQuant

| Aspect | OCTOPUS | PlanarQuant/IsoQuant |
|--------|---------|---------------------|
| Rotation | WHT (d×d, full) | Block-diagonal (2D/4D) |
| Encoding | Octahedral map + norm split | Per-coordinate Lloyd-Max |
| Bit allocation | Non-uniform (b+1, b-1) | Uniform per component |
| FMAs | 16,384 (WHT) | 256/512 (block) |
| Parameters | 16,384 | 128 |
| Norm handling | Separate ρ per triplet | Separate L2 norm per vector |
| Quality (synthetic) | Best MSE at all bits | Slightly higher MSE |
| Quality (real PPL) | Not benchmarked yet | Beats TurboQuant |
| Speed | Slow (full WHT) | Fast (block diagonal) |

### The Gap

OCTOPUS wins on synthetic MSE because its octahedral map + non-uniform bit split are MSE-optimal. But it uses the same O(d²) WHT rotation as TurboQuant. PlanarQuant/IsoQuant achieve comparable real-world quality with O(d) block rotations. **The ideal combination would be OCTOPUS's encoding scheme with PlanarQuant/IsoQuant's rotation scheme.**

## Verdict

**High-value integration target.** PlanarQuant and IsoQuant represent a fundamental speed improvement over all our current quantization backends (TurboQuant, SpectralQuant, OCTOPUS) because they replace O(d²) or O(d log d) rotation with O(d) block rotation.

Recommendation:
1. Add PlanarQuant as a new rotation backend — simplest, fastest, proven PPL
2. Add IsoQuant as a quality option at 4-bit — best quality at 4-bit, 4D blocks
3. Consider hybrid: OCTOPUS encoding + PlanarQuant/IsoQuant rotation (best of both)
4. Keep OCTOPUS for scenarios where best MSE matters (e.g., MaxSim scoring)
5. Feature gate as `planar_quant` / `iso_quant` (opt-in initially, GOAT to prove)

### Risk Assessment

- **Low risk**: Pure rotation replacement — same Lloyd-Max codebooks, same storage format
- **Proven results**: Real PPL on Llama 3.1 8B and Qwen2.5-3B, llama.cpp production integration
- **Clean implementation**: 2D Givens is ~50 lines of Rust, 4D quaternion ~80 lines
- **Inverse rotation required**: Must implement for V-cache (simple transpose for Givens)

### Why This Matters for GOAT

Our current OCTOPUS GOAT (Bench 022) uses WHT rotation. If we replace WHT with block-diagonal rotation, we get:
- 32-64× fewer FMAs during rotation
- 128× fewer rotation parameters to store
- Potentially faster encode/decode despite same Lloyd-Max quantization
- Real PPL parity or improvement (proven on real models)

The question: does OCTOPUS's octahedral encoding + non-uniform bit split still dominate with block-diagonal rotation? This is the GOAT proof we need.