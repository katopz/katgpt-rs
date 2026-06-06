# KV Cache Compression — Research & Alternatives

> Default production codec: **Hybrid OCT+PQ** (OCTOPUS triplet encoding + PlanarQuant 2D Givens rotation).
> See main README for the default GOAT stack. This document covers the full detail of all alternatives.

## Production Stack

**Hybrid OCT+PQ** — default-on, best MSE + best rotation cost (Bench 024, Plan 101). Combines OCTOPUS triplet encoding with PlanarQuant's 2D Givens rotation: equal-or-lower MSE, better MaxSim, 64× fewer rotation FMAs than pure OCTOPUS.

---

## 1. TurboQuant — Legacy Baseline

## 🗜️ TurboQuant: Near-Optimal KV Cache Compression (Legacy Baseline)

Legacy baseline for benchmarking and education. Superseded by **Hybrid OCT+PQ** (primary default, Plan 101) and **SpectralQuant** (calibrated alternative). Compresses KV cache from f32 (32 bits) to 2-4 bits per coordinate using random rotation + Lloyd-Max scalar quantization. Based on [TurboQuant (Zandieh et al., 2025)](https://arxiv.org/pdf/2504.19874).

| Metric | Flat f32 | TQ 3-bit | TQ 4-bit |
|--------|----------|----------|----------|
| Bytes/token | 128 | 24 (**5.3×**) | 24 (**5.3×**) |
| 32K ctx memory | 1073.7 MB | 151.0 MB (**7.1×**) | 151.0 MB (**7.1×**) |
| Key cosine sim | 1.0000 | 0.9825 | 0.9958 |
| Attention correlation | 1.0000 | 0.9907 | 0.9978 |
| Output cosine sim | 1.0000 | 0.9989 | 0.9975 |

Architecture: random orthogonal rotation → Beta-distributed coordinates → Lloyd-Max codebook → bit-packed storage. Unbiased attention scores by construction (E[estimated] = true).

**Zero-alloc hot path (Plan 051):** Pre-allocated scratch buffers eliminate all heap allocations from `store_key`/`store_value`/`dequantize_key_into`/`dequantize_value_into`. Full store+dequant cycle **44.6% faster**, per-call dequantize **17-20% faster** at production kv_dim.

📁 `src/turboquant/` — `codebook.rs`, `rotation.rs`, `kv_cache.rs`, `forward.rs`, `types.rs`
🔧 Feature flag: `turboquant` (off by default, legacy baseline)

---

## 2. SpectralQuant — Calibrated Eigenbasis

## 🔬 SpectralQuant: Calibrated Eigenbasis KV Compression (Secondary, Default-On)

Data-driven spectral analysis replaces TurboQuant's random rotation with a calibrated eigenbasis. Near-optimal quantization via offline calibration → water-fill bit allocation → Lloyd-Max codebooks. **Secondary KV compression** — useful for per-dimension water-fill adaptation (Plan 077). Superseded by OCTOPUS (primary default, zero calibration, -22% to -49% MSE vs SQ). At same 3-bit budget with real calibration (Bench 013): SQ cosine=0.9845 > TQ 0.9715, SQ MaxSim error=18.90% < TQ 40.54% (2.1× lower), SQ compression=9.7× > TQ 5.3×. SQ wins quality AND compression at matched budget vs TQ.

| Technique | What | Why Better Than TQ |
|-----------|------|--------------------|
| Eigenbasis rotation | Covariance → eigendecomposition | Rotates along data's natural axes, not random |
| Water-fill allocation | Per-dim bits ∝ eigenvalue | High-energy dims get more bits, low-energy get fewer |
| Two-regime quantization | Semantic (high-energy) + tail | Optimal non-uniform codebook per regime |
| Participation ratio | d_eff = (Σλ_i)² / Σ(λ_i²) | Measures intrinsic dimensionality — typically 4–6 at d_h=128 |

**Key properties:**
- **Calibrated once:** `SpectralQuantCalibration` computed offline per (layer, head, kv_type), serialized with model weights
- **Spectral gap detection:** λ_d_eff / λ_{d_eff+1} reveals when eigendecomposition captures most variance
- **Cumulative variance thresholds:** `var_95`, `var_99` — min components for 95%/99% energy retention
- **Zero-alloc hot path:** Same pre-allocated buffer strategy as TurboQuant

📁 `src/spectralquant/` — `types.rs`, `spectral.rs`, `nonuniform_quant.rs`, `spectral_rotation.rs`, `spectral_kv_cache.rs`, `forward.rs`
🔧 Feature flag: `spectral_quant` (**on by default**)

---

## 3. OCTOPUS — Octahedral Triplet Codec

## 🐙 OCTOPUS: Octahedral Triplet KV Cache Compression (Data-Oblivious, Legacy)

Data-oblivious triplet codec that beats calibrated SpectralQuant at all bit widths. Groups rotated coordinates into contiguous 3-blocks, encodes direction via octahedral map (S² → [-1,1]²), and applies MSE-optimal non-uniform bit split (b+1 for direction, b-1 for norm). Based on [OCTOPUS (Boss et al., 2026)](https://arxiv.org/abs/2605.21226).

**GOAT proof (Bench 022):** OCTOPUS vs SpectralQuant (calibrated, 256 samples) at d=128:

| Metric | SQ 2-bit | OCT 2-bit | SQ 3-bit | OCT 3-bit | SQ 4-bit | OCT 4-bit |
|--------|----------|-----------|----------|-----------|----------|-----------|
| MSE | 0.1233 | **0.0962** (-22%) | 0.0379 | **0.0263** (-31%) | 0.0145 | **0.0074** (-49%) |
| Cosine | 0.9368 | **0.9512** (+1.5%) | 0.9812 | **0.9870** (+0.6%) | 0.9930 | **0.9963** (+0.3%) |
| Calibration | 256 samples | **0 samples** | 256 samples | **0 samples** | 256 samples | **0 samples** |

**First data-oblivious codec to beat a calibrated codec in our benchmarks.** Joint 3×3 rounding gives additional 6-9% MSE reduction (encoder-only, zero decoder change).

**Production stack position:**
1. **Hybrid OCT+PQ** — **default-on**, best MSE + best rotation cost (Bench 024, Plan 101)
2. **OCTOPUS** — legacy baseline (same encoding, slower rotation; Bench 022/023)
3. **PlanarQuant** — speed fallback (per-coordinate quantization)
4. **SpectralQuant** — calibrated alternative, useful for per-dimension water-fill adaptation
5. **IsoQuant-Fast** — opt-in, 4D quaternion block rotation (32× fewer FMAs)
6. **TurboQuant** — legacy baseline (off by default)

📁 `src/octopus/` — `octahedral.rs`, `triplet.rs`, `codebook.rs`, `types.rs`, `encode.rs`, `kv_cache.rs`, `forward.rs`
🔧 Feature flag: `octopus` (pulled in by `hybrid_oct_pq`, in `full`)

---

## 4. PlanarQuant & IsoQuant — Block-Diagonal Rotation

## 🔧 Block-Diagonal Rotation: PlanarQuant & IsoQuant (Opt-In Speed Alternatives)

Block-diagonal rotation alternatives to OCTOPUS's full WHT. Replaces O(d²) rotation with O(d) per-block rotation for KV cache quantization. Based on [RotorQuant (Zandieh et al., 2025)](https://www.scrya.com/rotorquant.pdf).

| Backend | Rotation | FMAs (d=128) | Params | Quality |
|---------|----------|-------------|--------|---------|
| **PlanarQuant** | 2D Givens | 256 | 128 | MSE 0.034 (3-bit) |
| **IsoQuant-Fast** | 4D quaternion (left) | 512 | 128 | MSE 0.034 (3-bit) |
| TurboQuant/OCTOPUS | WHT (full) | 16,384 | 16,384 | MSE 0.034/0.026 (3-bit) |

**GOAT proof (Bench 023, d=128, 512 keys, 8 seeds):**

| Metric | PlanarQuant | IsoQuant-F | OCTOPUS | TurboQuant |
|--------|-------------|------------|---------|------------|
| MSE (3-bit) | 0.0340 | 0.0340 | **0.0265** | 0.0341 |
| Cosine (3-bit) | 0.9831 | 0.9831 | **0.9869** | 0.9831 |
| Rotation FMAs | **256** | 512 | 16,384 | 16,384 |
| Params | **128** | 128 | 16,384 | 16,384 |

**Key finding:** OCTOPUS's quality advantage comes from its octahedral triplet encoding, NOT rotation. PQ/IQ/TQ all cluster at MSE ≈ 0.034 with Lloyd-Max encoding. Block-diagonal rotation is sufficient — 64× fewer FMAs with <1% quality trade-off.

**Hybrid OCT+PQ (Bench 024):** Combining OCTOPUS triplet encoding with PlanarQuant's 2D Givens rotation is strictly better — equal-or-lower MSE, better MaxSim, 64× fewer rotation FMAs than pure OCTOPUS. Hybrid is the new production default.

📁 `src/planar_quant/` — `types.rs`, `rotation.rs`, `kv_cache.rs`, `mod.rs`
📁 `src/iso_quant/` — `types.rs`, `rotation.rs`, `kv_cache.rs`, `mod.rs`
🔧 Feature flags: `planar_quant` (opt-in), `iso_quant` (opt-in)

---

## 5. KVarN — Variance-Normalized KV Cache

## 📐 KVarN: Variance-Normalized KV Cache Compression (Default-On)

Variance normalization (Sinkhorn-style dual-scaling) + sub-channel group quantization for KV-cache compression. First method in the stack that directly targets **error accumulation in autoregressive decoding** — critical for reasoning/CoT workloads. The variance normalization is orthogonal to all existing methods and composable with SpectralQuant, Shard, and Plasma ternary.

**GOAT proof (Bench 053, Plan 179):** ALL PASS — promoted to default-on.

| Criterion | Measured | Target | Status |
|-----------|----------|--------|--------|
| 2-bit cosine | **0.9894** | ≥ 0.98 | ✅ |
| 4-bit cosine | **0.9979** | ≥ 0.98 | ✅ |
| Error accumulation ratio | **1.0116** | ≤ 1.5× | ✅ |
| Quantize overhead | **0.57%** | ≤ 1% | ✅ |
| Dequant overhead vs RTN | **+0.0%** | ≤ 2% | ✅ |

**KVarN vs RTN at 2-bit:** 0.9894 vs 0.9563 (+3.5% cosine). At 4-bit: parity (0.9979). KVarN dominates at the aggressive compression end.

**Key insight:** At 2-bit (4 quantization levels), variance normalization's dual-scale reconstruction compounds multiplicative errors. The fix: skip VarN at ≤2-bit, use sub-channel group quantization (group_size=4) instead — each group of 4 elements gets its own scale/zp, giving 32× more scales per tile but dramatically tighter quantization ranges.

**Production stack position:**
1. **Hybrid OCT+PQ** — default-on, best MSE + best rotation cost (Bench 024, Plan 101)
2. **KVarN** — default-on, best 2-bit quality + error accumulation resistance (Bench 053, Plan 179)
3. **SpectralQuant** — calibrated alternative, per-dimension water-fill adaptation
4. **OCTOPUS** — legacy baseline (same encoding, slower rotation)
5. **PlanarQuant/IsoQuant** — speed fallback (block-diagonal rotation)
6. **TurboQuant** — legacy baseline (off by default)

📁 `src/kvarn/` — `variance_norm.rs`, `kv_cache.rs`, `pseudo_decode.rs`, `mod.rs`
🔧 Feature flag: `kvarn` (**default-on**)

---

## 6. Asymmetric K/V Compression

## 🗜️ Asymmetric K/V Cache Compression (Plan 123, Research 081)

**Core finding:** V-side compression is quality-free while K precision is critical. Softmax amplifies K errors exponentially O(e^ε) but V errors only scale linearly O(w·ε). This is a mechanistic property of attention, not model-specific.

**GOAT proof (25/25 ✅):** All 24 proofs + cross-method benchmark pass (Bench 036).

| Config | key_bits | val_bits | cos_k | cos_v | combined | compression |
|--------|----------|----------|-------|-------|----------|-------------|
| symmetric (3,3) | 3 | 3 | 0.9910 | 0.9911 | 0.9910 | 10.67× |
| aggressive (8,2) | 8 | 2 | 1.0000 | 0.9581 | 0.9786 | 6.40× |
| **recommended (8,3)** | **8** | **3** | **1.0000** | **0.9910** | **0.9955** | **5.82×** |
| inverted (2,8) | 2 | 8 | 0.9579 | 1.0000 | 0.9785 | 6.40× |

**Recommended config:** `key_bits=8, val_bits=3` — near-perfect K reconstruction with <1% V quality loss. 5.82× compression. Asymmetric beats inverted at same bit budget because K fidelity matters more than V fidelity under softmax.

```rust
use katgpt_rs::types::AsymmetricKVConfig;

let config = AsymmetricKVConfig::default(); // key_bits=8, val_bits=3

// With TurboQuant (feature-gated)
let cache = TurboQuantKVCache::new_asymmetric(&config);
```

📁 `src/types.rs` — `AsymmetricKVConfig` · `src/benchmark.rs` — `bench_asymmetric_cross_method()` · `src/turboquant/kv_cache.rs` — `new_asymmetric()`
🔧 Feature flag: `asymmetric_kv` (opt-in, depends on `turboquant`)
