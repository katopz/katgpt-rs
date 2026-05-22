# GOAT 022: OCTOPUS Octahedral KV Cache Compression

**Date:** 2025-06-28 (updated 2025-07-05)
**Plan:** 099 (OCTOPUS Octahedral Triplet KV Cache)
**Command:** `cargo test -p microgpt-rs --features "octopus,spectral_quant,turboquant" --test bench_octopus_goat -- --nocapture`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, debug profile

## Configuration

- d ∈ {64, 128, 256}
- Nominal bits ∈ {2, 3, 4}
- OCTOPUS bit split: direction = b+1, norm = b-1
- 512 Gaussian keys, 64 Gaussian queries, 8 rotation seeds
- Joint 3×3 rounding enabled (default)
- SpectralQuant calibrated on 256 samples (realistic prefill calibration)

## 1. OCTOPUS vs SpectralQuant (Default) — d=128, 512 keys

> **Primary comparison.** SQ is default-on, calibrated. OCTOPUS is data-oblivious (zero calibration).

| bits | SQ MSE      | OCT MSE     | MSE Δ%      | SQ Cos    | OCT Cos   | Cos Δ% | Winner  |
|------|-------------|-------------|-------------|-----------|-----------|--------|---------|
| 2    | 0.123319    | **0.096215**| **-22.0%**  | 0.936769  | **0.951172** | +1.5% | OCTOPUS |
| 3    | 0.037891    | **0.026287**| **-30.6%**  | 0.981205  | **0.986978** | +0.6% | OCTOPUS |
| 4    | 0.014516    | **0.007436**| **-48.8%**  | 0.992954  | **0.996329** | +0.3% | OCTOPUS |

**Verdict: OCTOPUS dominates SpectralQuant at every bit width** — even without calibration data.
- At 2-bit: 22% MSE reduction, +1.5% cosine — OCTOPUS wins at extreme compression
- At 3-bit: 31% MSE reduction, +0.6% cosine — production-relevant, clear win
- At 4-bit: 49% MSE reduction, +0.3% cosine — quality gap widens with more bits

This is the **first data-oblivious codec to beat a calibrated codec** in our GOAT benchmarks.

## 2. Reconstruction Quality (↓ MSE, ↑ Cosine — better)

| d   | bits | MSE (mean)  | MSE (std)   | Cosine (mean) | IP Error | Eff. bpc |
|-----|------|-------------|-------------|----------------|----------|----------|
| 64  | 2    | 0.0990      | 0.00101     | 0.9503         | 1.989    | 2.333    |
| 64  | 3    | 0.0277      | 0.00033     | 0.9865         | 1.045    | 3.333    |
| 64  | 4    | 0.0080      | 0.00011     | 0.9961         | 0.560    | 4.333    |
| 128 | 2    | 0.0962      | 0.00110     | 0.9512         | 2.803    | 2.333    |
| 128 | 3    | 0.0265      | 0.00029     | 0.9869         | 1.466    | 3.333    |
| 128 | 4    | 0.0075      | 0.00010     | 0.9963         | 0.782    | 4.333    |
| 256 | 2    | 0.0981      | 0.00052     | 0.9501         | 4.007    | 2.333    |
| 256 | 3    | 0.0271      | 0.00009     | 0.9865         | 2.092    | 3.333    |
| 256 | 4    | 0.0081      | 0.00004     | 0.9960         | 1.140    | 4.333    |

**Key observations:**
- Cosine > 0.95 at all dimensions with just 2-bit nominal (2.33 effective bpc)
- Cosine > 0.98 with 3-bit (3.33 effective bpc)
- MSE and cosine are remarkably stable across dimensions (64→256), confirming data-oblivious property
- Very low MSE variance across rotation seeds (std ≈ 1% of mean)

## 3. OCTOPUS vs TurboQuant (Legacy) — d=128

> Historical reference only. TQ is demoted legacy baseline (off by default).

| bits | TQ MSE   | OCT MSE  | MSE Δ%   | TQ Cos   | OCT Cos  | Cos Δ% |
|------|----------|----------|----------|----------|----------|--------|
| 2    | 0.1790   | 0.0962   | **-46.3%** | 0.9048   | 0.9512   | **+5.1%** |
| 3    | 0.0886   | 0.0263   | **-70.3%** | 0.9552   | 0.9870   | **+3.3%** |
| 4    | 0.0512   | 0.0074   | **-85.5%** | 0.9760   | 0.9963   | **+2.1%** |

## 4. Joint 3×3 Rounding Ablation (d=128)

| bits | MSE (simple) | MSE (joint) | Δ%     | Cos (simple) | Cos (joint) | Δ%   |
|------|--------------|-------------|--------|---------------|-------------|------|
| 2    | 0.1053       | 0.0962      | -8.7%  | 0.9468        | 0.9512      | +0.5% |
| 3    | 0.0289       | 0.0263      | -8.9%  | 0.9857        | 0.9870      | +0.1% |
| 4    | 0.0080       | 0.0074      | -6.6%  | 0.9961        | 0.9963      | +0.0% |

**Joint rounding gives 6-9% MSE improvement** across all bit widths (encoder-only, zero decoder change).

## 5. Compression Ratio

### OCTOPUS Only

| d   | bits | Flat (B) | OCTOPUS (B) | Ratio  | Eff. bpc |
|-----|------|----------|-------------|--------|----------|
| 64  | 2    | 2048     | 192         | 10.7×  | 2.333    |
| 64  | 3    | 2048     | 256         | 8.0×   | 3.333    |
| 64  | 4    | 2048     | 320         | 6.4×   | 4.333    |
| 128 | 2    | 4096     | 336         | **12.2×** | 2.333 |
| 128 | 3    | 4096     | 464         | 8.8×   | 3.333    |
| 128 | 4    | 4096     | 592         | 6.9×   | 4.333    |
| 256 | 2    | 8192     | 640         | **12.8×** | 2.333 |
| 256 | 3    | 8192     | 896         | 9.1×   | 3.333    |
| 256 | 4    | 8192     | 1152        | 7.1×   | 4.333    |

### OCTOPUS vs TurboQuant (4 layers)

| d   | bits | Flat (B) | TQ (B) | OCT (B) | TQ Ratio | OCT Ratio |
|-----|------|----------|--------|---------|----------|-----------|
| 128 | 2    | 4096     | 288    | 336     | 14.2×    | 12.2×     |
| 128 | 3    | 4096     | 544    | 464     | 7.5×     | **8.8×**  |
| 128 | 4    | 4096     | 544    | 592     | 7.5×     | 6.9×      |
| 256 | 3    | 8192     | 1056   | 896     | 7.8×     | **9.1×**  |

## 6. Quality Across Dimensions (bits=2, most aggressive)

| d   | n_triplets | MSE     | Cosine  | IP Error |
|-----|------------|---------|---------|----------|
| 32  | 11         | 0.0904  | 0.9546  | 1.355    |
| 64  | 22         | 0.1004  | 0.9498  | 1.998    |
| 96  | 32         | 0.0949  | 0.9523  | 2.447    |
| 128 | 43         | 0.0968  | 0.9514  | 2.778    |
| 192 | 64         | 0.0970  | 0.9509  | 3.466    |
| 256 | 86         | 0.0973  | 0.9504  | 3.921    |

Quality is remarkably stable across dimensions (cosine 0.950-0.955).

## 7. Bit Split Sensitivity (d=128)

| dir_bits | nrm_bits | Total bits/triplet | MSE     | Cosine  |
|----------|----------|--------------------|---------|---------|
| 2        | 4        | 8                  | 0.2394  | 0.8748  |
| 3        | 3        | 9                  | 0.0968  | 0.9514  |
| **4**    | **2**    | **10**             | **0.0267** | **0.9869** |
| 5        | 1        | 11                 | 0.0075  | 0.9963  |

The **(b+1, b-1) = (4, 2) split** at nominal 3-bit gives the best quality per bit in the 10-bit budget range.

## 8. Production Stack Verdict

```
GOAT Production Stack (after Bench 022):
  1. OCTOPUS       — default-on, data-oblivious, dominates SQ at all bit widths (Bench 022)
  2. SpectralQuant — default-on, calibrated, still useful when per-dimension adaptation needed
  3. TurboQuant    — legacy, kept for backward compatibility only

Decision flow:
  if need_extreme_compression(bits <= 3):
      use Octopus         # -22% to -49% MSE vs SQ, no calibration needed
  elif need_per_dimension_adaptation():
      use SpectralQuant   # water-fill adapts to eigenvalue spectrum
  else:
      use Octopus         # default choice, better quality at all bit widths
```

### Quantitative Justification

| Metric (d=128) | SpectralQuant 2-bit | OCTOPUS 2-bit | Improvement |
|----------------|---------------------|---------------|-------------|
| MSE            | 0.1233              | 0.0962        | **22% ↓**   |
| Cosine         | 0.9368              | 0.9512        | **1.5% ↑**  |
| Calibration    | 256 samples         | 0 samples     | **Free**    |

| Metric (d=128) | SpectralQuant 3-bit | OCTOPUS 3-bit | Improvement |
|----------------|---------------------|---------------|-------------|
| MSE            | 0.0379              | 0.0263        | **31% ↓**   |
| Cosine         | 0.9812              | 0.9870        | **0.6% ↑**  |
| Calibration    | 256 samples         | 0 samples     | **Free**    |

| Metric (d=128) | SpectralQuant 4-bit | OCTOPUS 4-bit | Improvement |
|----------------|---------------------|---------------|-------------|
| MSE            | 0.0145              | 0.0074        | **49% ↓**   |
| Cosine         | 0.9930              | 0.9963        | **0.3% ↑**  |
| Calibration    | 256 samples         | 0 samples     | **Free**    |

**OCTOPUS is the first data-oblivious codec to beat a calibrated codec in our benchmarks.** Default-on as of Plan 099.

## Acceptance Criteria Status

- [x] `OctopusKVCache` implements `QuantizedKVCache` trait
- [x] All unit tests pass for octahedral encode/decode roundtrip
- [x] GOAT synthetic benchmark shows MSE improvement over SpectralQuant at d=128
- [x] Feature gate `octopus` works independently (`cargo test --features octopus`)
- [x] `SpKvQuantCache<OctopusKVCache>` compiles (composition proof — `test_sp_kv_octopus_composition_compiles` + `test_sp_kv_octopus_roundtrip` pass)
- [x] `.benchmarks/022_octopus_goat.md` populated with results
- [x] README updated with OCTOPUS section (T12)
- [x] OCTOPUS added to default features (GOAT proved: dominates SQ at all bit widths)