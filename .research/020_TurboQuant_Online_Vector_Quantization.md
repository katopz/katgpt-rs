# Research: TurboQuant — Online Vector Quantization with Near-Optimal Distortion Rate (20)

> Source: [TurboQuant](https://arxiv.org/pdf/2504.19874) by Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni (Google Research · NYU · Google DeepMind)
> Date: 2025-04, distilled 2025-07
> Raw code: `.raw/turboquant/`
> **Verdict: HIGH VALUE — KV Cache Compression for Production Inference, Direct Fit for microgpt-rs and riir-gpu**

## Summary

TurboQuant compresses high-dimensional vectors to low-bitwidth integers (1–4 bits per coordinate) while preserving geometric structure — both MSE (reconstruction quality) and inner products (attention score fidelity). It achieves near-optimal distortion, within a constant factor ≈ 2.7 of Shannon's information-theoretic lower bound, across all bit-widths and dimensions.

The method is **data-oblivious** (no calibration or k-means), **online** (applies instantly per token), and **accelerator-friendly** (vectorizable scalar quantization per coordinate). For KV cache quantization, it achieves quality neutrality at 3.5 bits per channel and marginal degradation at 2.5 bits per channel. For nearest neighbor search, it outperforms Product Quantization in recall with zero indexing time.

Core technique: **random orthogonal rotation → Beta distribution per coordinate → Lloyd-Max optimal scalar quantization**. For unbiased inner products (attention scores), a two-stage approach: MSE quantizer (b-1 bits) + QJL 1-bit residual sketch.

---

## Core Concepts

### MSE-Optimal TurboQuant (Algorithm 1)

For unit-norm vectors x ∈ S^{d-1}:

1. **Random rotation**: y = Π·x, where Π is a random orthogonal matrix (QR decomposition of random Gaussian). Rotated vector is uniformly distributed on the unit sphere.
2. **Beta distribution**: Each coordinate of y follows `f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)`, converging to N(0, 1/d) in high dimensions. Coordinates become nearly independent.
3. **Lloyd-Max codebook**: Solve continuous 1D k-means for 2^b centroids on the Beta distribution. Precompute once for (d, b) pairs.
4. **Quantize**: Find nearest centroid per coordinate via `searchsorted` on decision boundaries.
5. **Dequantize**: Look up centroids, rotate back via Π^T, rescale by stored norm.

**MSE bound** (Theorem 1): `D_mse ≤ (√(3π)/2) · 4^{-b}` for any b ≥ 0. For b=1,2,3,4: ≈ 0.36, 0.117, 0.03, 0.009.

### Inner-Product TurboQuant (Algorithm 2)

MSE quantizers introduce bias in inner product estimation (multiplicative bias 2/π at b=1). Solution:

1. Apply MSE quantizer at (b-1) bits → residual r = x - x̂_mse has small L2 norm.
2. Apply QJL (Quantized Johnson-Lindenstrauss) to residual: `sign(S · r)` where S has i.i.d. N(0,1) entries.
3. Store `||r||₂` for rescaling.

**Dequantized inner product**: `<y, x̂_mse> + ||r|| · √(π/2)/d · <y, S^T · qjl_signs>`

**Properties** (Theorem 2):
- **Unbiased**: E[<y, x̂>] = <y, x>
- **Distortion**: `D_prod ≤ (√(3π²)/2) · ||y||²/d · 4^{-b}` for any b ≥ 0

### Lower Bound (Theorem 3)

For any randomized quantizer Q: R^d → {0,1}^{b·d}:
- `D_mse ≥ 4^{-b}` (MSE lower bound from Shannon + Yao's minimax)
- `D_prod ≥ ||y||²/d · 4^{-b}` (inner product lower bound)

TurboQuant's MSE is within factor ≈ 2.7 of optimal. At b=1, factor drops to ≈ 1.45.

### Key Properties

| Property | Value |
|----------|-------|
| Data-oblivious | No calibration, no k-means, no data-dependent tuning |
| Online | Applies per-token during generation, no preprocessing |
| Accelerator-friendly | Scalar quantization per coordinate, fully vectorizable |
| Near-optimal | Within 2.7× of Shannon lower bound (1.45× at b=1) |
| Unbiased (Algorithm 2) | E[estimated <Q,K>] = true <Q,K> |
| Compression | f32 → b bits = 32/b× reduction (10.7× at 3 bits) |

---

## Experimental Results (from paper + .raw/turboquant/)

### KV Cache Quantization

| Model | Method | Bits | Needle-in-Haystack | LongBench Avg |
|-------|--------|------|--------------------|---------------|
| Llama-3.1-8B | Full precision | 16 | 0.997 | 50.06 |
| Llama-3.1-8B | TurboQuant | 3.5 | 0.997 | 50.06 |
| Llama-3.1-8B | TurboQuant | 2.5 | — | 49.44 |
| Llama-3.1-8B | KIVI | 3 | 0.981 | 48.50 |
| Llama-3.1-8B | PolarQuant | 3.9 | 0.995 | 49.78 |
| Llama-3.1-8B | SnapKV | 0.25× cache | 0.858 | — |

**3.5 bits = quality neutral. 2.5 bits = marginal degradation. 4.5× compression at 3.5 bits.**

### Nearest Neighbor Search

| Method | d=200 Recall@1 | d=1536 Recall@1 | Indexing Time |
|--------|---------------|-----------------|---------------|
| TurboQuant 4-bit | 0.72 | 0.93 | **0.001s** |
| PQ 4-bit (LUT256) | 0.58 | 0.82 | 240s |
| RabitQ 4-bit | 0.55 | 0.80 | 2268s |

TurboQuant achieves higher recall with **zero indexing time** vs hours for PQ/RabitQ.

### GPU Benchmark (from .raw/turboquant/ README)

**RTX 5090 — Qwen3.5-27B-AWQ (dense, 4-bit weights, TP=1)**:
- KV cache freed: **30.0 GB**
- Max token capacity: **2.0×** (457K → 914K)
- Decode throughput: +3.1% (TQ overhead negligible)
- Key compression cos_sim: **1.000000** (near-lossless at 3-bit)

**8× RTX 3090 — Qwen3.5-35B-A3B MoE (TP=8)**:
- KV savings: 30.9% of total KV (only compresses 10/40 full-attention layers)
- On pure dense transformer: would be **77% savings** (4.4× compression)
- Context extension: **1.45×** (1.4M → 2.0M tokens)
- Needle retrieval: **PASS** at all lengths (1K–131K)
- 5-needle coherence: **5/5** retrieved

### Adversarial Audit (from .raw/turboquant/)

| Claim | Verdict |
|-------|---------|
| "5.1× compression" | Misleading — honest: ~4.6× at 4K tokens, ~5× at 32K+ |
| "Near-lossless" | True for keys (cos_sim=1.0), values are bottleneck (cos_sim=0.94 at 2-bit) |
| "Zero indexing time" | True — data-oblivious, no preprocessing |
| "2× context on dense model" | True — measured 30 GB freed on Qwen3.5-27B |
| "1/4^b distortion scaling" | True — verified with unit-norm vectors |
| "Faster decode" | Within noise — TQ overhead is negligible, not a speedup |

---

## What Maps to Our System

### What Actually Applies

#### 1. KV Cache Compression for microgpt-rs (Highest Value, Direct Fit)

Our `MultiLayerKVCache` stores f32 keys and values in growing flat arrays. For long sequences, this is the memory bottleneck. TurboQuant at 3 bits gives ~10.7× compression with near-lossless quality.

Current code in `microgpt-rs/src/transformer.rs`:
- `MultiLayerKVCache`: flat `Vec<KVCache>` — **prime target**
- `PagedKVCache`: page pool — TQ compresses pages for longer context
- `RavenKVCache`: fixed slots — already conceptually compressed, less gain

The quantized KV cache would store:
- Bit-packed indices (b bits per coordinate)
- Per-token L2 norms (f32, negligible — 1 value per token vs d values)
- Per-layer codebook (2^b centroids, generated once from seed)

#### 2. GPU Attention Scoring Against Compressed Keys (High Value, riir-gpu)

Our `riir-ai/crates/riir-gpu/src/kernels/attention_score.wgsl` computes Q·K dot products in f32:

```wgsl
for (var d = 0u; d < params.head_dim; d = d + 1u) {
    dot = dot + query[d] * keys[k_base + d];
}
```

TurboQuant's inner-product mode computes attention scores **without dequantizing keys to f32**:
```
score = <Q, K̃_mse> + ||r|| · √(π/2)/d · <Q·Sᵀ, qjl_signs>
```

A new `attention_score_tq.wgsl` kernel would:
1. Gather centroids from bit-packed indices (codebook lookup)
2. Compute MSE contribution: dot product with centroid values
3. Compute QJL contribution: dot product with sign bits, scaled by residual norm
4. Result is unbiased — E[estimated score] = true score

This maps directly to our wgpu compute pipeline architecture.

#### 3. Raven RSM Synergy (Medium Value, Composable)

Plan 020's `RavenKVCache` already compresses KV to fixed slots via Top-K routing. TurboQuant compresses **within each slot**:
```
Raven: 32K tokens → 16 slots (2048× spatial compression)
TurboQuant: 16 slots × f32 → 3 bits (10.7× precision compression)
Combined: ~21,000× effective KV cache compression
```

This is composable — Raven handles the sequence dimension, TQ handles the embedding dimension.

#### 4. Nearest Neighbor for anyrag (Medium Value, Future)

anyrag uses embedding search for RAG retrieval. TurboQuant outperforms Product Quantization in recall with zero indexing time. Could replace PQ in the vector search layer:
- No k-means calibration during ingestion
- Better recall at same bit-width
- Data-oblivious — works immediately on new embeddings

#### 5. Lloyd-Max Codebook as Pure Rust Library (High Value, Foundation)

The codebook computation is self-contained math:
- Beta PDF on [-1, 1]
- Continuous 1D k-means (Lloyd-Max iterations)
- Precompute for (d, b) pairs, cache as JSON

This is ~200 lines of pure Rust using `statrs` for Gamma functions and numeric integration. No GPU needed. Forms the foundation for all other TQ work.

### What Does NOT Map

| TurboQuant Concept | Why It Doesn't Apply |
|---|---|
| **Triton kernels** | We use wgpu (WGSL), not CUDA Triton. Must write our own compute shaders. |
| **vLLM integration** | We have our own inference stack (microgpt-rs + riir-gpu). |
| **Outlier channel splitting** | Only needed at production scale with large models. Our draft model has head_dim=4 (too small). |
| **Entropy encoding of indices** | 5% gain at cost of complexity. Skip for now. |
| **Hybrid decode (paged + compressed)** | Our PagedKVCache is a different architecture. Would need separate integration. |
| **Value quantization at 2-bit** | Adversarial audit shows cos_sim=0.94 degradation. Use 3-4 bit for values. |

---

## Comparison: TurboQuant vs Our Existing Compression

| Aspect | Raven RSM (Plan 020) | Percepta (2D Hull) | PagedKVCache | TurboQuant (New) |
|--------|---------------------|--------------------|--------------|------------------|
| **Dimension** | Sequence (tokens→slots) | Spatial (2D hull) | Memory (pages) | Embedding (f32→bits) |
| **Compression axis** | N tokens → S slots | 2D points → hull vertices | Fixed-size pages | d floats → b·d bits |
| **Lossy?** | Yes (top-K routing) | No (exact hull) | No (paging) | Yes (near-optimal) |
| **Theoretical guarantee** | None formal | O(log N) hull size | Exact (paging) | Within 2.7× of Shannon bound |
| **Composable with others?** | Yes | Yes | Yes | **Yes** (orthogonal to all) |

**Key insight**: TurboQuant compresses along a dimension (embedding precision) that none of our existing methods touch. It's orthogonal to and composable with all of them.

---

## Application to Our System

### Direct Mappings

| Paper Concept | Our Equivalent | Status |
|---|---|---|
| **Random rotation Π** | New — QR decomposition of random matrix | ❌ Need to build |
| **Lloyd-Max codebook** | New — Beta distribution scalar quantizer | ❌ Need to build |
| **QJL projection S** | New — random Gaussian projection matrix | ❌ Need to build |
| **Bit-packed indices** | New — pack 2-4 bit indices into u8 | ❌ Need to build |
| **Attention scoring in quantized space** | `attention_score.wgsl` (currently f32) | ❌ Need new kernel |
| **KV cache storage** | `MultiLayerKVCache` (currently f32) | ❌ Need new variant |
| **Per-token norm storage** | Not tracked currently (implicit in f32) | ❌ Need to add |
| **Codebook precomputation** | New — offline computation, cached on disk | ❌ Need to build |

### What to Build (Gap Analysis)

#### Priority 1: Rust Lloyd-Max Codebook (Foundation)

Pure Rust implementation of Beta distribution scalar quantization:
- `statrs` for Gamma/log-Gamma functions
- Numeric integration (`quadrature` crate or manual trapezoidal)
- Lloyd-Max iteration until convergence
- Precompute for head_dim ∈ {4, 64, 128, 256} × bits ∈ {2, 3, 4}
- Cache as JSON or binary, load at startup

~200 lines, no GPU, no dependencies beyond `statrs`.

#### Priority 2: Random Rotation Matrix

QR decomposition of random d×d Gaussian matrix:
- `nalgebra` for QR decomposition
- Deterministic from seed (reproducible across runs)
- Store as `Vec<f32>` (d×d matrix, 64KB for d=128)

~50 lines.

#### Priority 3: TurboQuantKVCache in microgpt-rs

New KV cache variant alongside `MultiLayerKVCache`, `PagedKVCache`, `RavenKVCache`:
- Stores bit-packed indices + norms instead of f32 arrays
- Quantize on write (new token), dequantize on read (attention)
- Or: score directly in quantized space (Priority 4)

~300 lines in new file `microgpt-rs/src/turboquant.rs`.

#### Priority 4: attention_score_tq.wgsl in riir-gpu

New compute shader for GPU-accelerated attention scoring against compressed keys:
- Centroid gather from bit-packed indices
- QJL sign-bit contribution
- Unbiased inner product estimation

~100 lines WGSL + ~150 lines Rust binding.

#### Priority 5: Benchmark and Validation

Following the pattern of `bench_raven_vs_flat_cache`:
- `bench_turboquant_vs_flat_cache` — throughput comparison
- Quality metric: cos_sim between original and reconstructed KV
- Compression ratio: bytes stored / bytes original
- Attention score fidelity: correlation between f32 and quantized scores

---

## Key Takeaways

1. **Orthogonal compression axis.** TurboQuant compresses embedding precision (f32→bits). Our existing methods compress sequence length (Raven), spatial structure (Percepta), or memory layout (Paged). All are composable.

2. **Data-oblivious is a feature.** No calibration, no k-means, no data-dependent tuning. Works immediately on any model, any dimension. Zero indexing time. This matches our "config-driven" philosophy.

3. **Near-optimal is proven.** The 2.7× factor from Shannon's bound is not empirical — it's a theorem. For practical bit-widths (2-4), empirical results match theory closely.

4. **Unbiased attention scores matter.** Algorithm 2's inner product estimator has E[score] = true score. This means attention patterns are preserved on average — critical for generation quality.

5. **The reference implementation validates theorems.** `.raw/turboquant/` has 35 passing tests including theorem validation and adversarial audit. We can port from proven Python code.

6. **Values are the bottleneck, not keys.** The adversarial audit shows key compression is near-lossless (cos_sim=1.0 at 3-bit), but value compression degrades (cos_sim=0.94 at 2-bit). Use 3-4 bits for values, 2-3 bits for keys.

7. **Composable with Raven RSM.** Raven handles "which tokens to remember" (sequence compression). TurboQuant handles "how precisely to remember them" (precision compression). Combined: ~21,000× effective compression for the draft model.

8. **Complementary to Plan 042 (TTT Feedback Loop).** Plan 042 closes the training feedback loop. TurboQuant reduces the memory cost of inference, making longer contexts viable. Together: better models (Plan 042) + cheaper inference (this research).

9. **GPU kernel is the integration point.** The attention scoring kernel is where TurboQuant meets our inference pipeline. Replacing f32 dot products with quantized scoring is the single highest-impact change.

---

## Citation

```bibtex
@article{zandieh2025turboquant,
  title   = {TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate},
  author  = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal = {arXiv preprint arXiv:2504.19874},
  year    = {2025}
}
```
