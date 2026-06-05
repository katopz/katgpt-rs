# Research: Oscillatory State-Space Models — Modelless Distillation

**Date:** 2026-06-05
**Source:** arXiv 2606.02623 — "Oscillatory State-Space Models as Inductive Biases for Physics-Informed Neural PDE Solvers"
**Status:** GOAT Verdict Pending

---

## TL;DR

OSSM-PINN couples oscillatory state-space models (LinOSS) with fixed spectral bases to solve PDEs. The key insight: **eigenvalues on the imaginary axis** (not decaying) preserves structural information forever, unlike Mamba/S4/LRU which dampen. This is **modelless distillation gold** — we can apply the oscillatory prior to KV cache evolution, speculative decoding modes, and bandit frequency selection without any LLM training.

---

## Paper Core Ideas

### 1. LinOSS Cell — Forced Harmonic Oscillator

```python
# State: (y, z) ∈ R^H × R^H
# dy/dt = z                    (velocity)
# dz/dt = -A·y + B·s(t)        (acceleration = spring + forcing)
# A = ReLU(Ã) ≥ 0             (learnable frequencies)
# ω_i = √A_i                   (natural frequency per mode)
```

**Critical property:** Eigenvalues on imaginary axis ±i√A_i → **undamped oscillation**. Not Mamba's exponential decay. Not S4's HiPPO initialization. Pure energy-preserving oscillation.

### 2. Modal Factorization

```
û(x,t) = D(x) · Σ_k c_k(h(t)) · φ_k(x) + g(x)
```

- Spatial: fixed analytical basis (Fourier, Hermite, Pöschl-Teller)
- Temporal: learnable modal coefficients via LinOSS rollout
- **Decoupled:** swap basis without changing architecture

### 3. Parallel Scan — O(log N)

The recurrence `ξ_{n+1} = M·ξ_n + F_n` is **associative**:
```
(M2, F2) ∘ (M1, F1) = (M2·M1, M2·F1 + F2)
```

This means temporal rollout is parallelizable on GPU via Blelloch scan — O(log N) depth, not O(N) sequential.

### 4. Results That Matter

| Benchmark | Best Baseline | OSSM-PINN | Improvement |
|-----------|---------------|-----------|-------------|
| Convection β=100 | 3.2e-3 (NeuSA) | 1.5e-5 | **219×** |
| Wave | 5.5e-3 (NeuSA) | 4.2e-5 | **132×** |
| E-Bernoulli (extended) | 1.1e+0 (ML-PINN) | 1.5e-5 | **71,100×** |
| Schrödinger 100D | OOM (all) | 8.7e-4 | **Only method that fits** |

Loss landscape: L_log-Lip = 13.2 (OSSM) vs 37.9 (PINNsFormer) → **3× smoother optimization basin**.

---

## Distillation to katgpt-rs (Modelless)

### What NOT to Distill

- ❌ PDE solving (we're not solving PDEs — see Research 51)
- ❌ Physics-informed loss (no PDE residual to minimize)
- ❌ Spatial basis for PDE domains
- ❌ Boundary condition enforcement

### What TO Distill — Fundamental Principles

| Principle | Paper Mechanism | katgpt-rs Application |
|-----------|----------------|----------------------|
| **Oscillatory eigenvalues** | LinOSS: eigenvalues on imaginary axis | KV cache evolution that preserves structure instead of decaying |
| **Modal factorization** | Decouple spatial (fixed basis) from temporal (learned) | Decouple token embedding (fixed) from sequence evolution (learned modes) |
| **Parallel scan** | O(log N) temporal rollout | Already have Blelloch scan in HLA/Wall — reuse for LinOSS recurrence |
| **Learned frequencies** | ω = √(ReLU(Ã)) from residual | Bandit learns which oscillation frequencies capture temporal patterns in token streams |
| **Problem-adapted bases** | Fourier/Hermite/PT per domain | Per-domain spectral basis for speculative decoding (already have DomainLatent) |
| **Smooth loss landscape** | Oscillatory prior → flat minima | Better speculative decoding convergence in DDTree |

---

## Creative Fusion Ideas

### Fusion A: Oscillatory KV Cache (OscKV)

**Insight:** Current KV cache mechanisms are either:
- **Static:** Store everything, no evolution (standard attention)
- **Decaying:** Mamba/DeltaNet erase old information exponentially
- **Compressed:** SpectralQuant/Octopus compress via eigendecomposition

**Oscillatory alternative:** Let KV cache entries *oscillate* at learned frequencies per-head. For code/reasoning with cyclic patterns (loops, recursive calls, state machines), the cache should *revisit* information at the right frequency, not just decay it.

```rust
// Conceptual OscKV state per head
struct OscKVState {
    y: Vec<f32>,  // position (current value)
    z: Vec<f32>,  // velocity (trend)
    omega_sq: Vec<f32>, // learned frequency² per mode (H dims)
    beta: Vec<f32>,     // forcing projection
}
```

**Modelless:** ω² and β are learned via bandit from inference-time feedback (acceptance rate, latency), not LLM training. The bandit selects which oscillation frequencies to use per-query.

**Expected gain:** On cyclic/repetitive sequences (code generation, structured data), oscillatory cache revisits relevant context automatically. On one-shot queries, bandit routes to standard attention.

### Fusion B: Modal Speculative Decoding (ModalSpec)

**Insight:** Instead of drafting tokens autoregressively (one at a time), draft the **top-K Fourier modes** of the target sequence. The spatial basis is the token vocabulary embedding; the modal coefficients are the LinOSS rollout.

```
draft(tokens) = Σ_k c_k(h(t)) · φ_k(vocab)
```

Where:
- `φ_k(vocab)` = fixed Fourier basis over vocabulary embedding space
- `c_k(h(t))` = modal coefficient from LinOSS rollout
- `h(t)` = oscillatory state evolved from initial condition (prompt context)

**Modelless:** The Fourier basis over vocab embeddings is pre-computed once. The LinOSS rollout is a tiny linear recurrence (H=128). The bandit decides when to use ModalSpec vs standard DDTree.

**Expected gain:** Multi-token draft in O(log N) via parallel scan instead of O(N) autoregressive. Potentially 2-5× faster drafting for structured outputs.

### Fusion C: Frequency Bandit (FreqBandit)

**Insight:** The paper shows that **learned natural frequencies align with analytical dispersion** — the LinOSS automatically discovers the characteristic frequencies of the system.

**Application:** A bandit whose arms are frequency bands. Each arm captures a temporal pattern in the token stream:
- Low frequency: global context, document-level structure
- Medium frequency: paragraph-level coherence
- High frequency: local token-level patterns

The bandit learns which frequency band is most informative for the current query and routes to the appropriate decode strategy.

**Modelless:** Frequency bands are pre-defined. The bandit is pure inference-time. No LLM training.

**Expected gain:** Better speculative decoding accuracy with fewer draft tokens. Bandit learns "this query needs high-frequency drafting" vs "this query needs global context."

---

## GOAT Verdict

### Modelless Feasibility Assessment

| Fusion | Modelless? | Expected Gain | Risk | Verdict |
|--------|------------|---------------|------|---------|
| **A: OscKV** | ✅ Bandit-learned ω | Better cyclic pattern recall | Unclear if oscillation helps non-cyclic tasks | **Conditional GOAT** — gate behind `osc_kv` feature, validate on code benchmarks |
| **B: ModalSpec** | ✅ Pre-computed Fourier + LinOSS | 2-5× faster drafting | Complex integration, unproven for discrete tokens | **Experimental** — research prototype, not production |
| **C: FreqBandit** | ✅ Pure inference-time | Better decode strategy selection | Simple, low risk | **GOAT** — minimal implementation, high signal |

### Decision: Fusion C (FreqBandit) → GOAT, default on if no perf hurt

Fusion A and B require more research. Fusion C is immediately implementable and testable.

**FreqBandit is the simplest application of the oscillatory principle:** learn which temporal frequency band matters for each query, route decode strategy accordingly. The bandit arms are pre-defined frequency ranges. The reward is acceptance rate × latency improvement.

### Why FreqBandit Over Alternatives

1. **SOLID:** Single Responsibility — each arm handles one frequency band
2. **DRY:** Reuses existing BanditPruner infrastructure
3. **Modelless:** No LLM training, pure inference-time bandit
4. **Testable:** Before/after speculative decoding quality metrics
5. **Perf-safe:** If bandit selects wrong arm, falls back to standard decode

### Commercial Strategy Alignment

Per Research 003 (Commercial Open Source Strategy):
- **Engine (MIT):** FreqBandit lives in katgpt-rs, open source
- **Fuel (SaaS):** The learned frequency-arm mapping per domain is the fuel
- **No conflict:** This is inference-time optimization, not translation training

---

## Related Existing Infrastructure

| Existing Component | Location | Reuse Potential |
|-------------------|----------|-----------------|
| BanditPruner | `katgpt-core/traits.rs` | Direct reuse — FreqBandit is a bandit arm selector |
| ThinkingBanditFrozen | `src/` (thinking_cot feature) | Pattern: bandit learns when to think → same pattern for frequency selection |
| Blelloch parallel scan | HLA research (#28) | Reuse for LinOSS parallel rollout if we implement Fusion A/B |
| SpectralQuant eigendecomposition | `src/spectralquant/` | Eigendecomposition of KV cache covariance → basis for frequency analysis |
| DomainLatent | `katgpt-core/types.rs` | Per-domain learned embeddings → per-domain frequency profiles |
| InferenceRouter + TriggerGate | `src/` | Already routes CPU/GPU/ANE → add frequency-band routing dimension |

---

## What's Missing for Implementation

1. **FFT of token streams** — need to compute spectral content of recent token sequences to initialize frequency bands
2. **Bandit arm → decode strategy mapping** — which frequency band maps to which speculative decode config
3. **Before/after benchmarks** — need test cases showing cyclic vs non-cyclic query performance
4. **CPU/GPU auto-route integration** — FreqBandit should feed into TriggerGate for compute tier selection

---

## TL;DR

OSSM-PINN's oscillatory state-space models (eigenvalues on imaginary axis, not decaying) are a fundamental departure from Mamba/S4 dampening. The **modelless distillation** is a Frequency Bandit that learns which temporal frequency band (low/medium/high) matters for each query and routes decode strategy. This is the simplest application: GOAT, default on if no perf hurt. The deeper applications (OscKV, ModalSpec) are conditional — gate behind feature flags, validate before enabling.
