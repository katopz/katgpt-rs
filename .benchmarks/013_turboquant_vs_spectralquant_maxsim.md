# Benchmark 013: TurboQuant vs SpectralQuant MaxSim CPU Results

**Date:** 2025-01-25
**Plan:** 080 (MaxSim Late-Interaction Scoring)
**Command:** `cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## 1. MaxSim Core Primitive (T2 Correctness)

| Metric | Value |
|--------|-------|
| MaxSim score (Lq=8, Ld=32, dim=64) | 82.6837 |
| Naive reference | 82.6837 |
| Match | ✓ exact |

Per-query-token breakdown:
- q[0] → best doc[5] dot=64.0000 (strong signal match)
- q[1] → best doc[5] dot=5.1357
- q[2] → best doc[5] dot=6.2895
- q[3] → best doc[5] dot=4.9539
- q[4] → best doc[5] dot=1.6574
- q[5] → best doc[2] dot=0.1061
- q[6] → best doc[2] dot=0.2485
- q[7] → best doc[2] dot=0.2926

## 2. Packed Scoring (T2 Consistency)

| Pair | Query | Doc | Score |
|------|-------|-----|-------|
| 0 | q0 (128tok) | d0 (512tok) | 16.7615 |
| 1 | q0 (128tok) | d2 (384tok) | 78.7575 |
| 2 | q1 (256tok) | d0 (512tok) | 56.6013 |
| 3 | q1 (256tok) | d1 (192tok) | 17.9992 |

Packed matches sequential: ✓

## 3. Block Scoring: MaxSim vs Mean-K (T7 Quality)

| Method | Needle | Noise | Separation |
|--------|--------|-------|------------|
| MaxSim | 435.1998 | 21.7600 | **20.00×** |
| Mean-K dot | 2.8900 | 0.6800 | 4.25× |

**MaxSim separation: 4.71× better** at distinguishing needle from noise.

## 4. SIMD Speedup (T4 Performance)

| Metric | Value |
|--------|-------|
| Config | Lq=32, Ld=256, dim=128, 1000 iterations |
| MaxSim latency | 46.9 µs/call |
| Throughput | 21,321 scores/s |
| Naive latency | 353.4 µs/call |
| **Speedup** | **7.53×** |

## 5. TurboQuant MaxSim (T9 Correctness)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, 4-bit quantization |
| TurboQuant score | 18.9444 |
| Uncompressed score | 19.1255 |
| **Relative error** | **0.95%** |
| Threshold | < 10% |
| Status | ✓ PASS |

## 6. SpectralQuant MaxSim (T10 Correctness)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, ~3-bit spectral quantization |
| SQ MaxSim (streaming) | 16.9787 |
| SQ MaxSim (dequantized) | 16.9787 |
| **Roundtrip error** | **0.00%** (exact match) |
| Status | ✓ PASS |

Note: SQ uses identity eigenvectors → random rotation fallback (no real calibration).

## 7. TurboQuant vs SpectralQuant Head-to-Head

kv_dim=16, 10,000 iterations per timing run.

### Config: 8 doc positions × 2 query tokens

| Method | Score | Error vs Uncompressed | Latency | Bits/key |
|--------|-------|-----------------------|---------|----------|
| TurboQuant | 18.9444 | **0.95%** | **2.63 µs** | 4 |
| SpectralQuant | 16.9787 | 11.22% | 2.82 µs | ~3 |
| Uncompressed | 19.1255 | 0.00% | — | 32 |

### Config: 16 doc positions × 4 query tokens

| Method | Score | Error vs Uncompressed | Latency | Bits/key |
|--------|-------|-----------------------|---------|----------|
| TurboQuant | 38.3281 | **0.07%** | **10.22 µs** | 4 |
| SpectralQuant | 35.6501 | 7.05% | 11.06 µs | ~3 |
| Uncompressed | 38.3537 | 0.00% | — | 32 |

### Latency Comparison

| Config | TQ (µs) | SQ (µs) | SQ overhead |
|--------|---------|---------|-------------|
| 8pos×2q | 2.63 | 2.82 | +7.2% |
| 16pos×4q | 10.22 | 11.06 | +8.2% |

## GOAT Gate Summary

| Gate | Metric | Result | Status |
|------|--------|--------|--------|
| T2 | Correctness: naive within 1e-6 | exact match | ✅ |
| T4 | Speedup: ≥2× vs naive | **7.53×** | ✅ |
| T7 | Separation: ≥5% better than mean-K | **371% better** | ✅ |
| T9 | TQ error: < 10% vs uncompressed | **0.95%** | ✅ |
| T10 | SQ streaming vs dequantized | **0.00%** exact | ✅ |
| T15 | Example exercises all primitives | **7/7 sections** | ✅ |

## Analysis: TurboQuant vs SpectralQuant

### Without Calibration (identity eigenvectors → random rotation fallback)

- **TQ wins on quality:** 0.07–0.95% error vs SQ's 7.05–11.22% error
- **TQ wins on latency:** 7–8% faster (simpler uniform dequantize vs variable-bit unpack)
- **SQ uses 25% less storage:** ~3 bits vs 4 bits per element

### Why SQ appears worse here

This benchmark uses **identity eigenvectors** (no real calibration data). The bug fix in Plan 080 correctly detects this and substitutes a random rotation — identical to what TQ does. But SQ operates at **~3 bits** (avg_bits=3.0) while TQ uses **4 bits**. With the same rotation quality but fewer bits, SQ naturally has higher quantization error.

### When SQ should win (theoretical)

Per Research 39 ("3% Is All You Need: Breaking TurboQuant's Compression Limit"), SQ's advantage requires:
1. **Real calibration data** — eigenvectors from actual model K/V vectors during prefill
2. **Eigenbasis decorrelation** — concentrates variance into fewer dimensions
3. **Variable-bit allocation** — semantic dimensions get more bits, tail gets fewer
4. **Selective QJL correction** — sign correction in semantic regime (not yet implemented)

With real calibration: SQ at 3 bits should match or beat TQ at 4 bits (same quality, 25% less memory).

### Verdict

| Scenario | Winner | Why |
|----------|--------|-----|
| No calibration data | **TurboQuant** | More bits + simpler = better quality + lower latency |
| Real calibration data | **SpectralQuant** (theoretical) | Eigenbasis decorrelation + variable-bit should give same quality at 25% less storage |
| Latency-critical path | **TurboQuant** | 7–8% faster, simpler dequantize path |
| Memory-constrained path | **SpectralQuant** | ~3 bits vs 4 bits = 25% KV cache reduction |

### Implementation Status

- ✅ TQ: production-ready, 0.95% error at 4-bit
- ✅ SQ: bug-fixed, roundtrip exact, but no real calibration pipeline yet
- ⏳ SQ: needs `calibrate_eigenbasis()` wired with real model K/V vectors to achieve theoretical superiority
- ⏳ SQ: needs selective QJL correction implementation to fully utilize `b_high = b_low + 1` formula

## Test Commands

```sh
# Run all tests (683 pass)
cargo test --features "maxsim,turboquant,spectral_quant" --lib --quiet

# Run benchmark example
cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release

# Clippy clean
cargo clippy --features "maxsim,turboquant,spectral_quant" --examples --quiet
```
