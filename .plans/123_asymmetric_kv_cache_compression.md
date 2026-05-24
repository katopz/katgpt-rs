# Plan 121: Asymmetric K/V Cache Compression — GOAT Proof

> **Status:** 📋 Planned
> **Branch:** `develop/feature/121_asymmetric_kv`
> **Depends on:** Plan 043 (TurboQuant ✅), Plan 077 (SpectralQuant ✅), Plan 099 (OCTOPUS ✅), Plan 100 (PlanarQuant/IsoQuant ✅), Plan 101 (HybridOctPq ✅)
> **Research:** `.research/081_Asymmetric_KV_Cache_Compression.md`
> **Source:** [Asymmetric K/V Cache Compression](https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md) — Tom Turney
> **Feature gate:** `asymmetric_kv` (opt-in, depends on `turboquant`)
> **Goal:** Prove that V cache compression is quality-free while K precision is critical, establish asymmetric defaults across all KV cache methods via GOAT proofs and benchmarks.

## Summary

The attention mechanism's softmax amplifies K-side errors exponentially (O(e^ε)) while V-side errors scale linearly (w·ε). Our architecture already supports separate `key_bits`/`val_bits` in all 6 KV cache variants — we just haven't proven the asymmetric advantage.

This plan:
1. Adds GOAT proofs that V compression is quality-neutral and K compression is quality-critical
2. Benchmarks symmetric vs asymmetric across all compression methods
3. Establishes asymmetric defaults (`key_bits=8, val_bits=3`) as the recommended config
4. Documents the finding for all downstream consumers (riir-ai, modelless distillation)

## Why This Matters

- **Mechanistically proven** — softmax amplification is fundamental to attention, not model-specific
- **Independently validated** — 10+ researchers, 5 GPU backends, 3 quantization methods (TurboQuant, E8 lattice, PolarQuant)
- **Zero architecture cost** — our `key_bits`/`val_bits` separation already exists
- **SpectralQuant confirms** — key d_eff ≈ 4 (3% of d_h) vs value d_eff ≈ 40 (31% of d_h) explains the asymmetry mechanistically

## Tasks

- [ ] **T1: `asymmetric_kv` feature gate** — Cargo.toml
  - Add `asymmetric_kv = ["turboquant"]` feature
  - Gate all new benchmarks and proofs behind this feature
  - File: `Cargo.toml`

- [ ] **T2: Asymmetric benchmark helper** — `benchmark.rs`
  - `fn bench_asymmetric_kv<C: QuantizedKVCache>(config: &Config, key_bits: u8, val_bits: u8) -> AsymmetricBenchResult`
  - Measures: cosine_sim_key, cosine_sim_value, compression_ratio, store_latency, dequant_latency
  - Compares: (key_bits, val_bits) vs (val_bits, key_bits) — same total bits, swapped allocation
  - Returns struct with all metrics for GOAT assertion
  - File: `src/benchmark.rs`

- [ ] **T3: GOAT proof — V compression is free** — `benchmark.rs` tests
  - For each KV cache method (TQ, SQ, OCT, Hybrid, Planar, Iso):
    - Store random K/V vectors at val_bits=2 with key_bits=8
    - Assert: cosine_sim(original_v, dequant_v) > 0.98
    - Assert: compression_ratio > 2.5×
  - Gate: `#[cfg(feature = "asymmetric_kv")]`
  - File: `src/benchmark.rs` (test module)

- [ ] **T4: GOAT proof — K precision is critical** — `benchmark.rs` tests
  - For each KV cache method:
    - Store random K/V vectors at key_bits=2 with val_bits=8
    - Assert: cosine_sim(original_k, dequant_k) < 0.90 (degraded)
    - Compare with symmetric (key_bits=2, val_bits=2): equally bad
    - Compare with asymmetric (key_bits=8, val_bits=2): K quality restored
  - Proves: at same total bits, allocation to K vs V matters 10×+
  - Gate: `#[cfg(feature = "asymmetric_kv")]`
  - File: `src/benchmark.rs` (test module)

- [ ] **T5: GOAT proof — asymmetric beats symmetric at same budget** — `benchmark.rs` tests
  - Fix total_bits = key_bits + val_bits = 6 (e.g., 3+3 vs 5+1 vs 8+(-2 invalid, skip))
  - Configs: (3,3) symmetric, (4,2) mild asymmetric, (5,1) aggressive asymmetric (if valid)
  - Assert: (key_high, val_low) consistently beats (val_high, key_low) in combined cosine metric
  - Metric: `harmonic_mean(cosine_sim_k, cosine_sim_v)` — captures both sides
  - Gate: `#[cfg(feature = "asymmetric_kv")]`
  - File: `src/benchmark.rs` (test module)

- [ ] **T6: Cross-method asymmetric benchmark** — `benchmark.rs`
  - `fn bench_asymmetric_cross_method(config: &Config) -> Vec<MethodAsymmetricResult>`
  - For each method: symmetric (3,3) vs asymmetric (8,3) vs asymmetric (8,2)
  - Print table: method | config | cos_k | cos_v | compression | winner
  - Called from `run_all` when `asymmetric_kv` feature is active
  - File: `src/benchmark.rs`

- [ ] **T7: `AsymmetricKVConfig` type** — `types.rs`
  - `struct AsymmetricKVConfig { pub key_bits: u8, pub val_bits: u8 }`
  - `impl Default` → `key_bits: 8, val_bits: 3` (the safe default from paper)
  - `fn compression_ratio(&self, kv_dim: usize) -> f32` — theoretical compression
  - `fn is_asymmetric(&self) -> bool` → `self.key_bits != self.val_bits`
  - File: `src/types.rs`

- [ ] **T8: Update `TurboQuantKVCache` recommended constructor** — `turboquant/kv_cache.rs`
  - Add `pub fn new_asymmetric(config: &Config) -> Self` → `key_bits=8, val_bits=3`
  - Doc: "Recommended asymmetric config from Research 081. V compression is quality-free."
  - Keep existing `new(config, key_bits, val_bits)` for custom configs
  - File: `src/turboquant/kv_cache.rs`

- [ ] **T9: Benchmark result file** — `.benchmarks/035_asymmetric_kv_goat.md`
  - Auto-generated from T6 benchmark run
  - Table per method: symmetric vs asymmetric cosine sims + compression
  - GOAT proof pass/fail summary
  - File: `.benchmarks/035_asymmetric_kv_goat.md`

- [ ] **T10: Update README** — `README.md`
  - Add "🗜️ Asymmetric K/V Compression" section after TurboQuant section
  - Key finding: V compression is free, K precision is critical
  - Recommended config: `key_bits=8, val_bits=3`
  - GOAT proof reference
  - File: `README.md`

## Architecture

```
src/
  types.rs                    # T7: AsymmetricKVConfig
  benchmark.rs                # T2-T6: asymmetric benchmarks + GOAT proofs
  turboquant/kv_cache.rs      # T8: new_asymmetric() constructor

.benchmarks/
  035_asymmetric_kv_goat.md   # T9: auto-generated results

Cargo.toml                    # T1: asymmetric_kv feature gate
README.md                     # T10: documentation
```

## GOAT Proof Targets (8/8 ✅ target)

| # | Property | Assertion | Method |
|---|----------|-----------|--------|
| 1 | V-free at 2-bit | cos_v(8,2) > 0.98 for all methods | T3 |
| 2 | V-free at 3-bit | cos_v(8,3) > 0.99 for all methods | T3 |
| 3 | V-free at 4-bit | cos_v(8,4) > 0.995 for all methods | T3 |
| 4 | K-critical at 2-bit | cos_k(2,8) < 0.90 | T4 |
| 5 | K-critical dominates V-critical | cos_k(2,8) < cos_v(8,2) always | T4 |
| 6 | Asymmetric beats symmetric | combined(8,2) > combined(3,3) at same budget | T5 |
| 7 | Cross-method consistency | V-free holds for TQ+SQ+OCT+Hybrid+Planar+Iso | T3 |
| 8 | Compression meaningful | ratio(8,3) > 2.5× for all methods | T3 |

## Key Design Decisions

1. **Feature-gated, not default-on** — The benchmarks and proofs are opt-in (`asymmetric_kv`). The *finding* (use asymmetric) is a config recommendation, not a code change. Existing code works identically.

2. **Micro config validation first** — Our micro config (head_dim=4, n_kv_heads=1) is too small for GQA effects and weight-quantization stacking. We prove the *mechanistic* property (softmax amplification) at micro scale, which is model-independent.

3. **No new KV cache variant** — We don't create `AsymmetricKVCache`. Asymmetry is a *config* of existing variants, not a new compression method.

4. **Cosine similarity as proxy** — We can't measure PPL without a real model, but cosine similarity between original and dequantized vectors directly measures reconstruction fidelity. The paper's PPL findings follow from cosine fidelity.

5. **`new_asymmetric()` not `new_default()`** — The name makes the asymmetric intent explicit. Users who want symmetric can still call `new(config, 3, 3)`.

## Expected Results

Based on Research 081 and the paper's cross-hardware validation:

| Config | Expected cos_k | Expected cos_v | Expected Compression |
|--------|---------------|---------------|---------------------|
| (8, 3) asymmetric | ~1.000 | ~0.99+ | ~2.8× |
| (8, 2) asymmetric | ~1.000 | ~0.98+ | ~3.0× |
| (3, 3) symmetric | ~0.97 | ~0.97 | ~5.1× |
| (2, 2) symmetric | ~0.90 | ~0.90 | ~8.0× |
| (2, 8) inverted | ~0.90 | ~1.000 | ~3.0× |

The inverted (2,8) should have good V quality but terrible K quality — proving K matters more.

## Risks

1. **Micro config may not show strong asymmetry** — At head_dim=4, softmax is over very few positions. The amplification effect is weaker. Mitigation: test at multiple sequence lengths (8, 32, 128, 512) to find where asymmetry emerges.

2. **SpectralQuant's eigenbasis may reduce asymmetry** — SQ already exploits K/V d_eff difference. Mitigation: if SQ shows less asymmetric gain, that's a valid finding — SQ's spectral rotation partially addresses what asymmetric config addresses naively.

3. **OCTOPUS triplet encoding confounds** — OCT's (b+1, b-1) split between direction and norm is a different kind of asymmetry. Mitigation: test with uniform (b, b, b) split as well to isolate the K/V effect.

## Timeline

- T1-T2: Feature gate + benchmark helper (foundation)
- T3-T5: GOAT proofs (core value)
- T6-T7: Cross-method benchmark + config type
- T8: Constructor convenience
- T9-T10: Results + documentation