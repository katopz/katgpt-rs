# FlashLib: GPU Classical ML Kernel Tricks — Distillation & Verdict (Revised)

**Date:** 2026-05-28
**Source:** https://flashml-org.github.io/ (Yang et al., 2026)
**Code:** https://github.com/FlashML-org/flashlib
**Local mirror:** `.raw/flashlib/`
**Supersedes:** Initial R130 verdict (NO GAIN → PARTIAL GAIN after code audit)

---

## Summary

FlashLib is a GPU library (Triton + CuteDSL, NVIDIA-only) that accelerates classical ML operators with up to 26×–208× speedups over cuML on H200. After auditing the **actual source code** (not just the paper), four algorithmic kernel tricks transfer to our wgpu stack:

1. **PCA Dual-Gram Routing** — `N < 4*D → compute X·Xᵀ (N×N) instead of Xᵀ·X (D×D)` — **direct steal for SpectralQuant calibration**
2. **Roofline Cost Prediction** — predict runtime/FLOPs/bandwidth in ~5µs CPU-only — **pure Rust port, ~200 lines**
3. **x²-Free Fused Scoring** — skip ‖x‖² term, compute `-2⟨x,y⟩` directly — **for future clustering ops**
4. **Shape-Heuristic Kernel Routing** — BN/BM tile sizes auto-selected by shape — **our GemvAutotune already does this, FlashLib validates approach**

---

## Code Audit: What We Found

### Steal #1: PCA Dual-Gram Routing → SpectralQuant Calibration

**FlashLib code** (`primitives/pca/triton/pca.py` L73-116):
```python
def triton_pca(X, K, *, tol=None):
    N, D = X.shape
    if N >= 4 * D:
        return _triton_pca_cov(X, K, tol=tol)    # Xᵀ·X → (D×D)
    return _triton_pca_dual(X, K, tol=tol)        # X·Xᵀ → (N×N)
```

**Our code** (`spectralquant/calibration.rs` L289):
```rust
// ALWAYS computes d_h × d_h covariance, even when seq_len < d_h
let [dx, dy, dz] = dispatch_2d(d_h as usize, d_h as usize, 16, 16);
```

**The problem:** For short sequences (common in game AI — combat rounds, dialog turns), seq_len might be 16-128 while d_h is 128-256. We compute a 256×256 covariance matrix when a 32×32 Gram matrix would suffice. The eigendecomposition is O(n³), so this is a **(256/32)³ = 512× slowdown** on the eigen step.

**Transfer:** Add dual-Gram routing to `calibration.rs` — when `seq_len < 4 * d_h`, compute `X·Xᵀ` (seq_len × seq_len) instead of `Xᵀ·X` (d_h × d_h), then project eigenvectors back.

**Gain:** Up to **512× speedup** on SpectralQuant calibration for short sequences. Calibration is the bottleneck for KV cache compression — this directly speeds up model startup for game sessions.

### Steal #2: Roofline Cost Prediction → Inference Planning

**FlashLib code** (`info/roofline.py` L269-327):
```python
def roofline(flops, bytes_moved, dtype, device, op_type="gemm", ...):
    t_compute_ms = (flops / 1e12 / eff_compute_tf) * 1000.0
    t_memory_ms = (bytes_moved / 1e12 / eff_bw_tbs) * 1000.0
    t_launch_ms = max(n_launches, 1) * LAUNCH_OVERHEAD_MS  # 50µs
    runtime_ms = max(t_compute_ms, t_memory_ms, t_launch_ms)
```

Pure stdlib, no GPU import, ~200 lines. Calibrated throughput table:
```python
_SUSTAINED_TFLOPS = {
    ("gemm", "fp32", "H200"): 50.0,
    ("kmeans", "bf16", "H200"): 400.0,
    ("knn_build", "bf16", "H200"): 600.0,
}
```

**Our code:** `spec_cost_model` (Amdahl for speculative only), `GemvAutotune` (runtime benchmark, ~100ms per shape), `sr2am_configurator` (UCB1 planning). No unified cost surface.

**Transfer:** Port `roofline.py` → `roofline.rs` in katgpt-core. Add Apple M-series hardware peaks (we already benchmark extensively). Calibrate from existing benchmark data.

**Gain:** SR²AM + SpecHop + MaxSim can predict cost before dispatching. Enables "should I use GPU or CPU for this batch?" decisions in ~5µs instead of ~100ms autotune. **This is the agent-native API FlashLib bragged about.**

### Steal #3: x²-Free Fused Distance → Future Clustering

**FlashLib code** (`primitives/knn/impl.py` L24):
```python
# Never materialises an N×M cross matrix to HBM and never loads x_sq
# x²-free score: ‖x-y‖² = ‖x‖² + ‖y‖² - 2⟨x,y⟩
# Skip the ‖x‖² and ‖y‖² terms when only ranking matters
```

**Our code:** `maxsim_score.wgsl` computes raw dot products (already optimal for max-dot). But if we add KNN/DBSCAN-style nearest-neighbor for game AI (clustering game states, NPC behavior grouping), the x²-free trick saves one full buffer allocation per call.

**Transfer:** Future — when we add clustering operations to riir-gpu.

**Gain:** 2× memory savings (no distance matrix materialization), ~1.5× speedup for KNN/DBSCAN operations.

### Steal #4: Shape-Heuristic Kernel Routing → Validates GemvAutotune

**FlashLib code** (`primitives/knn/cost.py` L87-106):
```python
def _heuristic_BN(B, N, M, K):
    NB = N * B
    if NB >= 50_000: return 128       # build regime
    if NB >= 30_000: return 64
    if NB <= 8:       return 8        # small-Q Pattern-A
    if NB <= 32:      return 16
    if NB <= 128:     return 32
    return 64
```

**Our code:** `GemvAutotune` already benchmarks Plane vs Tiled variants per (m, n) pair with cached results.

**Verdict:** FlashLib validates our approach. No steal needed — we're already doing it. FlashLib uses heuristic tables (no autotune), we use runtime benchmarking. Both are correct for their platform (CUDA has predictable performance; Metal/Vulkan has more device variance so autotune is safer).

---

## Verdict: PARTIAL GAIN — 2 Steals Worth Implementing

| Steal | Domain | Gain | Effort | Feature Gate | GOAT Proof | Super-GOAT? |
|-------|--------|------|--------|-------------|-----------|-------------|
| Dual-Gram Routing | SpectralQuant calibration | Up to 512× for short seq | ~2 days (WGSL + Rust) | `dual_gram_pca` (default-on after GOAT) | Calibration accuracy matches | If game AI benefits: **YES** |
| Roofline Cost Model | Inference planning | ~5µs vs ~100ms for cost prediction | ~1 day (pure Rust port) | `roofline_cost` (default-on) | Predicted vs actual within ±20% | Indirect — enables better SR²AM |
| x²-Free Distance | Future clustering | 2× memory, 1.5× speed | Future | N/A | N/A | No |
| Shape-Heuristic Routing | Validates existing | Validation only | 0 days | N/A | N/A | No |

### Why Steal #1 (Dual-Gram) Could Be Super-GOAT

FlashLib reports **PCA 47× over cuML** and **TruncatedSVD 208× over cuML**. The 208× figure is from their dual-Gram + Halko combination on wide-data shapes. Our SpectralQuant calibration does the exact same computation (covariance + eigendecomposition) for every (layer, head) at model load time.

**Game AI angle:** MMO game sessions are short (combat rounds ~16-64 tokens, dialog turns ~8-32 tokens). SpectralQuant calibration for short sequences would benefit massively from dual-Gram routing. If this makes KV cache compression calibration fast enough to do **per-game-session** (instead of per-model-load), that's a Super-GOAT selling point: "Personalized KV compression per combat encounter."

**Keep secret:** The specific ratio thresholds (when to switch from cov to dual-Gram) calibrated on game workloads.

### Why Steal #2 (Roofline) Is Default-On Material

Our SR²AM configurator already makes planning decisions. Adding a roofline cost model means it can predict cost without benchmarking. This is a quality-of-life improvement for the configurator, not a new capability. But it enables agents to compose pipelines with cost budgets — FlashLib's "agent-native API" concept.

---

## Plan

→ [Plan 159: FlashLib Dual-Gram Routing + Roofline Cost Model](../.plans/159_flashlib_dual_gram_roofline.md)

---

## Reference

```bibtex
@misc{yang2026flashlib,
  title  = {FlashLib: Bringing Flash Magic to Classical Machine Learning Operators},
  author = {Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Mang, Qiuyang and
            Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Gonzalez, Joseph E. and
            Han, Song and Xu, Chenfeng and Stoica, Ion},
  year   = {2026},
  url    = {https://flashml-org.github.io/},
}
```
