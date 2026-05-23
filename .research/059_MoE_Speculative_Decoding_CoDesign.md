# Research 59: MoE + Speculative Decoding Co-Design

> **Source:** [Why MoE models get more from speculative decoding](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) — Cohere (2026-04)
> **Paper:** [MoESD](https://arxiv.org/pdf/2505.19645) — MoE Speculative Decoding analysis
> **Date:** 2026-07, distilled 2026-07
> **Related Research:** 02 (Speculative Decoding), 06 (Raven RSM), 08 (Sparse MLP TwELL), 09 (EMO), 22 (Lighthouse Attention)
> **Related Plans:** 022 (Sparse MLP), 044 (PFlash), 026 (Inference Budget), 055 (MTP Drafter)
> **Verdict: PARTIAL VALUE — Three distillations: (1) Amdahl decomposition for LeviathanVerifier cost model, (2) batch-size-aware sparse MLP sparsity threshold, (3) Raven slot routing overlap metric. Core MoE routing findings do NOT apply (no MoE architecture). The arithmetic intensity framework and co-design principle are conceptual validation of our existing approach.**

---

## TL;DR

Cohere validates that MoE models benefit from speculative decoding differently than dense models, with three key mechanisms:

1. **Non-monotonic speedup curve**: MoE's low arithmetic intensity (k/N ratio) creates a "sweet spot" at moderate batch sizes where verification is nearly free because all experts are already loaded.

2. **Temporal correlation in expert routing**: Adjacent tokens route to overlapping experts (38% overlap at step 1 vs 6.3% uniform baseline), reducing unique expert loading during verification by 20-31%. This is a structural property of MoE, not workload-dependent.

3. **Fixed-overhead amortization at BS=1**: Non-expert operations (attention, norms, kernel launches) are overhead-dominated at BS=1. Verifying K+1 tokens in one pass amortizes these overheads, giving 2.18× speedup at BS=1 vs 1.87× predicted by expert routing alone.

**The co-design principle**: For a given target batch size, the sparsity ratio (k/N) and shared-to-routed expert ratio can be co-optimized to stay in the bandwidth-bound regime where SD delivers best returns.

**Our mapping**: We don't have MoE architecture, but we have analogous concepts:
- Sparse MLP (unstructured sparsity) ≈ MoE expert activation
- Raven RSM slot routing ≈ MoE expert routing (top-k selection)
- LeviathanVerifier ≈ Their target model verification
- Domain inference budget ≈ Their batch-size-aware co-design

---

## 1. Key Findings from the Article

### 1.1 Three Batch-Size Regimes for MoE SD

| Regime | BS Range | Characteristic | SD Benefit |
|--------|----------|----------------|------------|
| Low BS | 1-4 | Partial expert loading, bandwidth-bound | Limited by extra expert loading |
| Moderate BS | 4-32 | All experts loaded, bandwidth-bound | **Peak** — verification nearly free |
| High BS | 32+ | Compute-bound | SD gains vanish |

The key equation for arithmetic intensity:

```
AI = T × (k + S) / (N + S)

T = tokens, k = active experts/token, N = total experts, S = shared experts
```

Lower k/N → lower AI → stays bandwidth-bound longer → wider sweet spot.

### 1.2 Temporal Correlation in Expert Routing

Cohere measured expert overlap on their production MoE (k=8, N=128):

| Token Distance | Empirical Overlap | Independence Baseline | Uniform Baseline |
|----------------|-------------------|----------------------|------------------|
| Step 1 | 38.1% | 11.8% | 6.25% |
| Step 2 | 32.9% | 11.8% | 6.25% |
| Step 3 | 30.1% | 11.8% | 6.25% |
| Step 4 | 29.9% | 11.8% | 6.25% |

Per-layer variation: mid-network layers show strongest overlap (~50%), early layers (~32%), final layers (~25%).

**Result for SD verification (K=3)**: Only 20.36 unique experts needed empirically vs 25.4 predicted by independence baseline. That's 2.55× top-k, not the 3.2-3.6× baselines predict.

This holds across 13 Spec-Bench categories (0.377-0.385 overlap) and 7 languages. It's a structural property of natural text routing, not workload-dependent.

### 1.3 Amdahl's Law Decomposition

At BS=1, the target forward pass decomposes into:
- **Routed expert weight loading** (fraction f = 0.30 with shared experts)
- **Everything else** (fraction 1-f = 0.70): attention, norms, communication, kernel launches

```
T(K+1) / T(1) = f × unique(K+1)/k + (1-f)

For f=0.30, unique(4)/k=2.55:
T(4)/T(1) = 0.30 × 2.55 + 0.70 = 1.46× (Amdahl estimate)
Actual measured: 1.25×
```

The overestimate is because Amdahl ignores:
1. Non-expert GPU kernels are launch-overhead-dominated at BS=1 (2-10μs where 2-3μs is just launch)
2. Expert GEMMs with 4 tokens get better compute-read overlap

### 1.4 Co-Design Principle

For system designers:

| Target BS | Optimal Sparsity | Shared Expert Ratio |
|-----------|-----------------|---------------------|
| High BS | Lower k/N (sparser) | More routed, fewer shared |
| Low BS | Shared experts beneficial | More shared reduces verification cost |

Shared experts reduce verification cost at low BS but raise effective k/N, pushing compute-bound transition earlier.

---

## 2. What We Already Have (Conceptual Alignment)

| Cohere Concept | Our Equivalent | Location | Match |
|----------------|---------------|----------|-------|
| MoE top-k expert routing | Raven RSM: top-k slot routing | `transformer.rs` `raven_update` | ✅ Structural analog |
| Sparse expert activation | Sparse MLP: index packing for alive neurons | `microgpt-core/types.rs` `sparse_matmul` | ✅ Unstructured analog |
| Speculative decoding verification | `LeviathanVerifier` | `speculative/verifier.rs` | ✅ Direct match |
| Draft model (small, fast) | `Config::draft()` / `Config::bpe_draft()` | `types.rs` | ✅ Direct match |
| Arithmetic intensity (k/N) | Sparsity ratio in `sparse_matmul` alive count | `microgpt-core/types.rs` | 🟡 Analogous concept |
| Batch-size regimes | Domain inference budget (`tree_budget`, `beta`) | riir-ai Plan 026 | 🟡 Config-level only |
| Temporal correlation | Raven slot reuse across positions | Not measured | ❌ Gap (T1 below) |
| Amdahl decomposition | Not modeled for LeviathanVerifier | Not implemented | ❌ Gap (T2 below) |
| Co-design knobs | `InferenceOverrides` from TOML | riir-ai Plan 026 | 🟡 Partial |

---

## 3. Verdict: What Applies to Our Stack

### 3.1 What Does NOT Apply

| Finding | Why Not |
|---------|---------|
| MoE routing overlap (20-31% reduction) | We have no MoE router. Our Sparse MLP is unstructured sparsity (ReLU zeros), not structured expert routing. Tokens don't "select" experts. |
| Expert popularity skew | No learned routing function. Sparsity pattern is data-dependent (which ReLU neurons activate), not router-dependent. |
| Shared vs routed expert trade-off | No shared experts concept. Our MLP is either dense or sparse, no hybrid. |
| Non-monotonic speedup curve | Our models are too small for bandwidth-bound vs compute-bound regimes to matter at the system level. Our bottlenecks are different (allocation, cache misses, branch prediction). |

### 3.2 What DOES Apply (Conceptual Validation)

1. **Sparse verification is cheaper** — Cohere proves that if tokens share computation (experts/slots/active neurons), speculative verification costs less than naive multiplication. Our `sparse_matmul` already exploits this for single-token decode. The insight validates our approach.

2. **Temporal locality exists in activation patterns** — Adjacent tokens likely activate similar ReLU neurons (our unstructured sparsity analog). This hasn't been measured, but the structural argument holds: semantically similar tokens → similar hidden states → similar ReLU activation patterns.

3. **Fixed-overhead amortization matters at small scale** — Cohere shows kernel launch overhead (2-10μs) dominates at BS=1. Our CPU inference has analogous overheads: allocation, cache cold-start, branch misprediction. The LeviathanVerifier batching K+1 tokens amortizes these.

4. **Co-design principle** — The idea that sparsity level and inference parameters should be co-optimized for the target workload is exactly what our domain inference budget system does (Plan 026).

### 3.3 Honest Assessment

**We cannot implement MoE-specific optimizations because we have no MoE.** Our `sparse_mlp` is unstructured (which neurons are zero after ReLU), not structured (which experts a router selects). The mechanisms are fundamentally different:

```
MoE routing:     token → W_router → softmax → top-k expert indices → load expert weights
Our sparse_mlp:  token → W1 × input → ReLU → count non-zeros → skip zeros in W2 × hidden
```

MoE routing is a **learned selection function** applied to **separate weight blocks**. Our sparsity is an **emergent property of ReLU** applied to a **single weight matrix**. The Cohere findings about expert overlap and routing correlation don't transfer.

---

## 4. Distillations (3 Concrete Tasks)

### D1: Raven Slot Routing Overlap Metric (Conceptual, no feature gate)

**What**: Measure how often adjacent tokens route to the same Raven RSM slots. This is our closest analog to MoE expert routing overlap.

**Why**: If Raven shows temporal correlation in slot routing (like MoE shows in expert routing), it validates that our O(1) slot memory captures locality, and informs whether speculative verification can skip slot updates for adjacent tokens.

**How**: Add a diagnostic function that counts unique slots across K+1 consecutive tokens during LeviathanVerifier operation.

**Scope**: Diagnostic only, no architectural change. Feature-gate behind existing `domain_latent` (which gates Raven usage).

**Expectation**: Raven top-k routing is sigmoid-based (not softmax-based like MoE), so overlap may differ. But the structural argument (semantically similar tokens → similar routing) should hold.

### D2: Amdahl Cost Model for LeviathanVerifier (Conceptual, feature gate: `spec_cost_model`)

**What**: Implement the Amdahl decomposition from the article:

```
T_verify(K+1) / T_decode(1) = f_sparse × unique_ratio + (1 - f_sparse)

f_sparse = fraction of forward pass that scales with verification tokens
unique_ratio = how many "unique" sparse operations are needed vs single-token
```

**Why**: Currently we have no cost model for LeviathanVerifier. We can't predict how K (draft length) affects verification cost. The Amdahl framework gives us a principled way to:
- Estimate optimal K for a given model/config
- Compare sparse_mlp vs dense verification cost
- Inform domain inference budget (Plan 026) K selection

**How**: Instrument `LeviathanVerifier::speculate()` to measure:
1. Time in sparse/dense MLP operations (f_sparse)
2. Time in attention, norms, sampling (1-f_sparse)
3. Ratio of unique active neurons across K+1 tokens

**Scope**: New feature gate `spec_cost_model`. Off by default. Produces diagnostic output, doesn't change inference behavior.

**Benefit**: Enables data-driven K selection instead of hardcoded defaults.

### D3: Batch-Size-Aware Sparse MLP Threshold (Enhancement, no new feature gate)

**What**: The article shows that sparsity benefits depend on batch size regime. Our `sparse_matmul` currently uses a fixed sparsity threshold. Make it aware of the "batch size" (number of tokens being verified in Leviathan mode).

**Why**: At K+1 tokens during verification, if adjacent tokens share active neurons (temporal locality), the effective sparsity per token after the first is lower (more overlap → fewer new unique neurons to process). The sparse matmul should account for this:
- For K=3 verification: process first token normally, then for subsequent tokens, only compute the delta (newly active neurons not already loaded).

**How**: This is an optimization for the `LeviathanVerifier` verification loop. When processing K+1 tokens sequentially:
1. Track active neuron set across tokens
2. For subsequent tokens, only compute new neurons not in the set
3. Accumulate output from shared + delta neurons

**Scope**: Enhancement to `sparse_matmul` in `microgpt-core`. No new feature gate — enhancement of existing `sparse_mlp` feature.

**Note**: This is speculative. The benefit depends on actual temporal correlation in ReLU activation patterns, which we haven't measured. D1 (overlap metric) should be done first to validate the assumption.

---

## 5. Integration Points

| Component | Current | With D1 (Overlap) | With D2 (Cost Model) | With D3 (Delta Sparse) |
|-----------|---------|-------------------|----------------------|----------------------|
| `sparse_matmul` | Process all alive neurons per token | N/A (diagnostic) | Provides f_sparse | Process only delta neurons |
| `LeviathanVerifier` | Fixed K, no cost awareness | Reports slot overlap | Reports cost decomposition | Uses delta sparse matmul |
| Raven RSM | O(1) slots, top-k routing | Reports routing overlap | N/A | N/A |
| Domain inference budget | Static K from TOML | N/A | Dynamic K based on cost model | N/A |
| Feature gate | `sparse_mlp`, `domain_latent` | `domain_latent` | `spec_cost_model` (new) | `sparse_mlp` |

---

## 6. What This Does NOT Prove

1. **Does not prove we should add MoE.** At our model sizes (n_embd=16-64), MoE overhead (router, expert weights, load balancing) exceeds benefit. Dense + sparse + LoRA is more efficient.

2. **Does not prove our sparse_mlp behaves like MoE.** Unstructured sparsity (ReLU zeros) and structured sparsity (expert routing) are different mechanisms with different properties. Cohere's overlap measurements don't transfer.

3. **Does not prove delta sparse matmul will help.** The temporal correlation assumption (D3) is unvalidated for ReLU activations. It may be that adjacent tokens have low neuron overlap, making delta processing overhead exceed benefit.

4. **Does not prove non-monotonic speedup in our system.** Our bottleneck isn't memory bandwidth vs compute (the MoE regime). Our bottleneck is CPU efficiency (allocation, cache, SIMD utilization). The batch-size regimes don't apply at our scale.

---

## 7. Priority Assessment

| Distillation | Value | Effort | Risk | Priority |
|-------------|-------|--------|------|----------|
| D1: Raven overlap metric | Medium — validates locality assumption | Low — diagnostic | None — read-only | **P1** (do first) |
| D2: Amdahl cost model | Medium — enables optimal K selection | Medium — instrumentation | Low — diagnostic | **P2** (depends on D1) |
| D3: Delta sparse matmul | Low-Medium — only if overlap is high | High — new matmul variant | Medium — may not help | **P3** (only if D1 shows overlap > 30%) |

**Recommended approach**: Do D1 first. If Raven slot overlap is < 20%, stop (no locality to exploit). If overlap is 20-40%, do D2 to quantify the benefit. Only do D3 if D1+D2 show clear benefit.

---

## 8. Related Work in Our Stack

| Research | Connection |
|----------|-----------|
| Research 02 (Speculative Decoding) | Foundational — Leviathan et al. algorithm we already implement |
| Research 06 (Raven RSM) | O(1) slot routing — our analog to MoE expert routing |
| Research 08 (Sparse MLP TwELL) | Unstructured sparsity — our analog to MoE expert activation |
| Research 09 (EMO) | Document-level MoE routing — concluded "no MoE for us", same verdict here |
| Research 26 (Gemma MTP) | Multi-token prediction — alternative to SD for token generation |
| Research 39 (SpectralQuant) | KV cache compression — orthogonal to SD optimization |
| Plan 022 (Sparse MLP) | Our TwELL implementation — D3 enhances this |
| Plan 026 (Inference Budget) | Domain-specific K and budget — D2 informs this |
| Plan 044 (PFlash) | Block-sparse speculative prefill — complementary optimization |
| Plan 055 (MTP Drafter) | Multi-token draft — the "small model" in SD |
| Plan 096 (MoE+SD Co-Design) | This distillation — Raven overlap + Amdahl cost model |

---

## 9. Conclusion

Cohere's analysis is excellent systems work for production MoE models. The three mechanisms (arithmetic intensity regimes, temporal routing correlation, fixed-overhead amortization) are well-characterized with clear mathematical frameworks.

For our stack, the value is **conceptual validation + two small diagnostics**:

1. **Conceptual validation**: Our existing architecture (sparse_mlp + Raven routing + LeviathanVerifier + domain inference budget) aligns with the co-design principles Cohere identifies. We're already doing the right things at our scale.

2. **D1 (Raven overlap metric)**: Low-effort diagnostic that validates whether our slot routing has the temporal locality that makes MoE SD efficient. If yes, it opens the door to D2 and D3.

3. **D2 (Amdahl cost model)**: Medium-effort instrumentation that gives us data-driven K selection for LeviathanVerifier. Useful regardless of MoE applicability.

4. **D3 (Delta sparse matmul)**: High-effort optimization that should only be attempted if D1 shows significant temporal correlation. Currently speculative.

**The honest verdict**: Cohere's findings are about MoE models at 100B+ parameter scale running on GPUs. Our models are 3 orders of magnitude smaller running on CPU. The specific mechanisms don't transfer. But the **principles** (exploit locality, amortize overheads, co-design sparsity with workload) do, and we're already following them.