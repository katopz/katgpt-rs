# Benchmark 013: TurboQuant vs SpectralQuant MaxSim CPU Results

**Date:** 2025-01-25
**Plan:** 080 (MaxSim Late-Interaction Scoring)
**Command:** `cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## Corrigendum

Initial version of this benchmark compared **4-bit TurboQuant** vs **~3-bit SpectralQuant** with **identity eigenvectors** (no calibration). This was an unfair comparison — different bit budgets and SQ degraded to random rotation fallback. The corrected benchmark below uses the **same 3-bit budget** with **real calibration data** from `calibrate_eigenbasis`, matching the methodology of `tests/bench_spectralquant.rs::bench_spectralquant_cosine_vs_turboquant`.

## 1. MaxSim Core Primitive (T2 Correctness)

| Metric | Value |
|--------|-------|
| MaxSim score (Lq=8, Ld=32, dim=64) | 82.6837 |
| Naive reference | 82.6837 |
| Match | ✓ exact |

## 2. SIMD Speedup (T4 Performance)

| Metric | Value |
|--------|-------|
| Config | Lq=32, Ld=256, dim=128, 1000 iterations |
| MaxSim latency | 48.3 µs/call |
| Throughput | 20,688 scores/s |
| Naive latency | 356.8 µs/call |
| **Speedup** | **7.38×** |

## 3. Block Scoring: MaxSim vs Mean-K (T7 Quality)

| Method | Needle | Noise | Separation |
|--------|--------|-------|------------|
| MaxSim | 435.1998 | 21.7600 | **20.00×** |
| Mean-K dot | 2.8900 | 0.6800 | 4.25× |

**MaxSim separation: 4.71× better** at distinguishing needle from noise.

## 4. TurboQuant MaxSim (T9 Correctness, 4-bit)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, 4-bit quantization |
| TurboQuant score | 18.9444 |
| Uncompressed score | 19.1255 |
| **Relative error** | **0.95%** |
| Status | ✓ PASS |

## 5. SpectralQuant MaxSim (T10 Correctness, ~3-bit)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, ~3-bit spectral quantization |
| SQ MaxSim (streaming) | 16.9787 |
| SQ MaxSim (dequantized) | 16.9787 |
| **Roundtrip error** | **0.00%** (exact match) |
| Status | ✓ PASS |

## 6. Fair Head-to-Head: Same 3-bit Budget, Real Calibration

kv_dim=16, 3-bit budget, 16 positions, 2 query tokens.
Calibration via `calibrate_eigenbasis` from synthetic keys with exponential eigenvalue decay.
Calibration quality: d_eff=4.78, var_95=8, var_99=13.

### Results

| Metric | TurboQuant (3-bit) | SpectralQuant (3-bit) | Delta |
|--------|--------------------|-----------------------|-------|
| Key cosine | 0.9715 | **0.9845** | **+0.0129** |
| MaxSim error | 27.15% | **3.88%** | **-23.27%** |
| Compression | 5.3× | **9.7×** | **+83%** |
| MaxSim latency | **5.17 µs** | 5.69 µs | +10% |
| Uncompressed score | 12.8597 | 12.8597 | — |

### Cross-validation with bench_spectralquant test

The existing `tests/bench_spectralquant.rs::bench_spectralquant_cosine_vs_turboquant` confirms the same conclusion at kv_dim=16 (debug build):

| Metric | TurboQuant (3-bit) | SpectralQuant (3-bit) | Delta |
|--------|--------------------|-----------------------|-------|
| Key cosine | 0.9692 | **0.9917** | **+0.0225** |
| Value cosine | 0.9827 | **0.9917** | **+0.0089** |
| Compression | 5.3× | **9.1×** | **+72%** |

### Verdict

| Dimension | Winner | Evidence |
|-----------|--------|----------|
| **Cosine similarity** | **SpectralQuant ✓** | +0.01 to +0.02 higher cosine |
| **MaxSim fidelity** | **SpectralQuant ✓** | 7× less error (3.88% vs 27.15%) |
| **Compression** | **SpectralQuant ✓** | 83% more compression (9.7× vs 5.3×) |
| **Latency** | **TurboQuant** | 10% faster (5.17 vs 5.69 µs) |

**SpectralQuant dominates at same bit budget with real calibration.**

## 7. Why the Initial Comparison Was Wrong

The first version of Section 7 showed TQ winning because:

1. **Different bit budgets**: TQ at 4-bit vs SQ at ~3-bit — TQ had 33% more bits
2. **No calibration**: Identity eigenvectors caused SQ to fall back to random rotation (same as TQ), eliminating its eigenbasis advantage
3. **Result**: "4-bit TQ" vs "3-bit TQ with extra steps" — TQ naturally won

With fair comparison (same bits, real calibration):
- SQ's eigenbasis rotation concentrates variance into fewer dimensions
- SQ's two-regime bit allocation spends bits where they matter
- SQ achieves higher fidelity AND better compression simultaneously

## GOAT Gate Summary

| Gate | Metric | Result | Status |
|------|--------|--------|--------|
| T2 | Correctness: naive within 1e-6 | exact match | ✅ |
| T4 | Speedup: ≥2× vs naive | **7.38×** | ✅ |
| T7 | Separation: ≥5% better than mean-K | **371% better** | ✅ |
| T9 | TQ error: < 10% vs uncompressed | **0.95%** (4-bit) | ✅ |
| T10 | SQ streaming vs dequantized | **0.00%** exact | ✅ |
| T15 | Example exercises all primitives | **7/7 sections** | ✅ |

## Confirms Existing Decision

`Cargo.toml` default features include `spectral_quant` and exclude `turboquant` (labeled "legacy baseline for bench/educate only"). This benchmark proves that decision was correct:

- **SpectralQuant**: higher quality, better compression, default-on ✓
- **TurboQuant**: simpler, slightly faster, useful as baseline/comparison ✓

## Test Commands

```sh
# Run all tests (683 pass)
cargo test --features "maxsim,turboquant,spectral_quant" --lib --quiet

# Run benchmark example
cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release

# Run dedicated SQ vs TQ cosine comparison
cargo test --features "spectral_quant,turboquant" --test bench_spectralquant bench_spectralquant_cosine_vs_turboquant -- --nocapture

# Clippy clean
cargo clippy --features "maxsim,turboquant,spectral_quant" --examples --quiet
```
