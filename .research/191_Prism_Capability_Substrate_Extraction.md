# Research 191: Prism — Capability Substrate Extraction for Inference-Time Sparse Routing

> **Paper:** [Prism: Unlocking Language Model Capability Extraction](https://github.com/e-xperiments/prism-capability-extraction) — Abhishek Mishra, Krishna Pagare, CAISc 2026
> **Date:** 2026-06
> **Related Research:** 053 (CNA Steering), 008 (TwELL Sparse MLP), 038 (Domain Latent), 117 (GKD), 036 (ROPD), 085 (Deep Manifold), 175 (ThoughtFold), 176 (VortexFlow)
> **Related Plans:** 022 (Sparse MLP), 087 (CNA Steering), 216 (SubstrateGate — if gain)
> **Commercial Alignment:** Engine/Fuel split (Verdict 003) — Prism masks are engine infrastructure, collimated LoRA adapters are fuel

---

## TL;DR

Prism proves that a single LLM capability (arithmetic, translation, function calling) can run through **~5% of MLP channels** while the rest of the network is deactivated. The key mechanism is **collimation** — a behavior-preserving LoRA adapter that relocates capability into fewer channels without changing outputs. Pre-attribution collimation moves the sparsity frontier (29% → 91% recovery at same budget); post-attribution collimation packs behavior through fixed channels (19% → 85%).

**Fundamental insight for modelless inference**: Capabilities are separable in MLP channel space. Pre-computed capability masks (from CNA or ReLP attribution) can gate MLP computation at inference time, combining with existing ReLU activation sparsity for dual sparsity. This is **not** training — masks are static data loaded once per model.

**Verdict: ADOPT — SubstrateGate is the modelless fusion. Pre-computed capability masks × existing `sparse_mlp` × DDTree branch routing = capability-level speculative decoding. Direct extension of CNA (Plan 087) and Sparse MLP (Plan 022). No LLM training. GOAT-gated, default-on if proven.**

---

## Paper Core Contributions

### 1. Extraction Contract

A substrate is sufficient only if the capability survives when the complement is deactivated:

```
R(mk, θ) = score(Mθ under mk) / score(M unmasked)
```

Three deactivation operators:
- **Zero-isolation**: set unkept channels to 0 before down-projection (function calling)
- **Mean-ablation**: replace with dataset-mean activation (translation)
- **Counterfactual patching**: replace with matched counterfactual input (arithmetic)

### 2. Collimation — Basis Change, Not Compression

A LoRA adapter trained with behavior-preserving constraint (KL leash to base model). Key properties:
- NOT fine-tuning (output map held fixed)
- NOT pruning (no weights removed)
- NOT compression (effective rank RISES, 9.878 → 11.789)
- IS a basis change — rotates representation so capability concentrates into fewer channels
- Mask's share of MLP activation energy: 0.329 → 0.462 (concentrated, not compressed)

### 3. Collimation Frontier — Timing Matters

| Regime | Order | What Changes | Effect |
|--------|-------|-------------|--------|
| Pre-attribution | collimate → attribute → mask | Which channels carry capability | Frontier moves (smaller circuit) |
| Post-attribution | attribute → mask → collimate | How behavior routes through fixed channels | Intra-budget packing (IoU 0.92) |

### 4. Key Results

| Capability | Model | Before | After | MLP % |
|-----------|-------|--------|-------|-------|
| Arithmetic | Qwen2.5-Math-1.5B | 29.00% | 91.33% | 5.0% |
| Function calling | Qwen3-8B | 19.1% | 84.6% | 36.2% |
| Translation rescue | HY-MT1.5-1.8B | — | 87.5% | 81.0% |

### 5. ReLP Attribution

Single-pass linearised backward scoring of MLP channels:
- Exact forward pass, single linear backward pass
- Scores each MLP intermediate channel by relevance to teacher-forced log-probability
- No individual channel ablation needed
- One score `s(c)` per channel, sum absolute relevance over continuation positions

### 6. Controls — Why Collimation is Specific

- No-KL LoRA: 49.53% at compact, needs ~14× more channels for same recovery
- Full SFT: 90.80% but at ~14× more channels
- Random same-size mask: ≤ 2.2% recovery
- KL-constrained rank-32 LoRA: 91.33% at 5% — the only combination that works

---

## Fusion Architecture: SubstrateGate (Modelless)

### The Core Idea

Prism shows capabilities are sparse in MLP channel space. We already have:
- `sparse_mlp` — skips dead ReLU neurons (activation sparsity)
- `cna_steering` — discovers which neurons matter per behavior (capability discovery)

The fusion: **Dual sparsity = activation sparsity ∩ capability sparsity**

```
                   ┌─────────────────────────────────────┐
                   │        MLP Forward Pass              │
                   │                                      │
  input ──► w1 ──► ReLU ──► [activation mask (sparse_mlp)]
                                    │
                                    ∩  (intersection)
                                    │
                              [capability mask (substrate_gate)]
                                    │
                                    ▼
                              alive ∩ capable
                                    │
                                    ▼
                               w2 (down-proj)
                                    │
                                    ▼
                                 output
```

The capability mask is pre-computed offline (via CNA or ReLP attribution) and stored as a per-capability bitmask. At inference:
1. Classify the input task (embedding similarity or rule-based)
2. Load the corresponding capability mask
3. Apply intersection with ReLU activation mask
4. Only compute through the intersection

### SubstrateRouter Trait

```rust
/// Routes inputs to capability substrates at inference time.
/// Pre-computed masks loaded from offline CNA/ReLP attribution.
pub trait SubstrateRouter: Send + Sync {
    /// Select the capability mask for this input.
    /// Returns None if no specialized substrate exists (falls back to full MLP).
    fn select_mask(&self, tokens: &[u32], config: &Config) -> Option<&SubstrateMask>;
    
    /// Register a pre-computed capability mask.
    fn register_mask(&mut self, capability: &str, mask: SubstrateMask);
}

/// A sparse MLP channel mask for a named capability.
/// Bitmask over [layers × d_ff] MLP intermediate channels.
#[derive(Clone, Debug)]
pub struct SubstrateMask {
    pub capability: String,
    pub bits: Vec<u64>,       // packed bitmask, 1 = active channel
    pub layer_counts: Vec<u16>, // per-layer active channel count
    pub recovery: f32,        // measured recovery score (from offline testing)
    pub total_channels: usize, // total channels in the mask
}
```

### DDTree Fusion: Capability-Routed Speculative Substrates

The creative fusion: each DDTree branch can be routed through a different capability substrate:

```
DDTree Exploration
    │
    ├── Branch 1: "standard library mapping" → SubstrateMask("py_stdlib")
    │       └── ConstraintPruner validates Rust syntax
    ├── Branch 2: "async/await transform"    → SubstrateMask("async_transform")
    │       └── ConstraintPruner validates Rust syntax
    └── Branch 3: "error handling convert"   → SubstrateMask("error_handling")
            └── ConstraintPruner validates Rust syntax

Best branch selected by: logprob × recovery_score × constraint_validity
```

This is capability-level speculative decoding:
- Each branch only activates its capability's sparse substrate → faster
- Multiple substrates explored in parallel → better coverage
- ConstraintPruner validates each → correct output
- Recovery score from Prism extraction contract → quality guarantee

### Recovery as Screening Signal

Prism's recovery metric becomes a `ScreeningPruner` relevance score:

```rust
impl ScreeningPruner for SubstrateScreeningPruner {
    fn relevance(&self, token: u32, context: &PruningContext) -> f32 {
        // How much of the capability's behavior survives under this mask
        // Sigmoid-gated (not softmax) per project conventions
        sigmoid(self.mask.activation_concentration(context.hidden))
    }
}
```

### CPU/GPU Auto-Route

| Hardware | Strategy | Rationale |
|----------|----------|-----------|
| CPU | Sparse index-packed matmul with dual mask | Cache-friendly, per-branch low FLOPs |
| GPU | Batch multiple substrates, batched matmul | Amortize kernel launch across substrates |
| Auto-switch | When `n_branches × substrate_size > threshold` → GPU | GPU overhead only worth it above threshold |

---

## Mapping to Existing Architecture

### What We Already Have (No Gain — Infrastructure Exists)

| Prism Concept | Our Equivalent | Status |
|--------------|----------------|--------|
| MLP channel attribution | CNA contrastive neuron discovery (Plan 087) | ✅ GOAT proved, default-on |
| Sparse MLP execution | `sparse_mlp` with index packing (Plan 022) | ✅ Default-on |
| Channel activation masking | `active_indices` / `active_values` in ForwardContext | ✅ Working |
| Screening by relevance | `ScreeningPruner::relevance()` trait | ✅ Working |
| Constraint validation | `ConstraintPruner::is_valid()` trait | ✅ Working |
| Speculative branch exploration | DDTree with branch scoring | ✅ Working |
| Domain conditioning | `domain_latent` mid-layer injection | ✅ Default-on |

### What Prism Adds (Potential Gain)

| Prism Concept | Gap in Our System | Assessment |
|--------------|-------------------|------------|
| **Per-capability MLP channel masks** | CNA discovers neurons per behavior but doesn't persist them as reusable masks for sparse execution | **HIGH GAIN** — extends CNA from discovery to deployment |
| **Extraction contract (recovery test)** | No sufficiency testing — we know which neurons matter but not whether they're sufficient alone | **HIGH GAIN** — quality guarantee for sparse deployment |
| **Dual sparsity (activation ∩ capability)** | `sparse_mlp` only uses ReLU dead neurons, doesn't combine with capability masks | **HIGH GAIN** — intersection is smaller than either alone |
| **Pre-computed mask loading** | No infrastructure for loading/serving per-capability masks at inference | **MEDIUM GAIN** — engineering, not research |
| **Substrate-routed DDTree branches** | DDTree branches all use full MLP; no per-branch capability routing | **HIGH GAIN** — novel fusion, capability-level speculation |
| **Recovery as screening signal** | `ScreeningPruner` uses CNA relevance but not recovery under mask | **MEDIUM GAIN** — extends existing trait |

### What Doesn't Apply (Modelless Constraints)

| Prism Concept | Why Not Modelless | Alternative |
|--------------|-------------------|-------------|
| Collimation training (LoRA) | Requires LLM training → riir-ai domain | Pre-computed masks from offline CNA attribution |
| ReLP backward pass | Requires model forward/backward at attribution time | CNA forward-only discovery (already implemented) |
| Merge-scale sweep | Training-time procedure | Not needed — masks are static |
| KL-constrained training | Training-time constraint → riir-ai domain | Collimated adapters are fuel (riir-ai produces, katgpt-rs consumes) |

---

## GOAT Gate Criteria

For `substrate_gate` to be default-on:

| Gate | Criteria | Rationale |
|------|----------|-----------|
| G1 | Accuracy ≥ 98% of baseline on benchmark tasks | Capability mask must not degrade quality |
| G2 | Throughput ≥ 100% of baseline (no perf hurt) | Sparse execution must not add overhead |
| G3 | FLOPs ≤ 60% of baseline for single-capability tasks | Measurable FLOPs reduction |
| G4 | CNA-discovered masks achieve ≥ 50% of Prism recovery | Our forward-only attribution must be useful |
| G5 | DDTree substrate routing improves acceptance rate ≥ 5% | Capability routing must help exploration |
| G6 | Feature disabled = zero overhead | `#[cfg(feature = "substrate_gate")]` guard |
| G7 | All existing tests pass with/without feature | No regression |

**If all gates pass: `substrate_gate` becomes default-on.**

---

## Performance Model

### Expected FLOPs Reduction

Prism shows 5% MLP channels sufficient for arithmetic after collimation. Our CNA discovers 0.1% neurons per behavior. The intersection:

| Sparsity Source | Typical % | After Intersection |
|----------------|-----------|-------------------|
| ReLU dead neurons | 50-90% alive | 50-90% |
| CNA capability mask | 0.1-5% | 0.05-4.5% |
| **Combined** | — | **0.05-4.5% of full MLP** |

At 4.5% MLP channels:
- Dense MLP FLOPs: 2 × n_embd × d_ff = 2 × 4096 × 16384 = 134M FLOPs
- Sparse MLP FLOPs: 2 × 4096 × 737 = 6M FLOPs
- **22× FLOPs reduction in MLP**

But MLP is ~67% of decode FLOPs, so total decode reduction:
- 67% × 0.05 = 3.4% of total (best case, 4.5% substrate)
- 67% × 0.20 = 13.4% of total (moderate, 20% substrate)

Realistic expectation: **10-40% total decode FLOPs reduction** for capability-targeted workloads.

### Hot-Path Concerns

1. **Mask intersection cost**: O(d_ff) per layer per token — negligible vs matmul
2. **Cache pressure**: Capability mask bits in L1 — `L × d_ff / 64` bytes per mask
   - For L=32, d_ff=16384: 32 × 256 = 8KB — fits in L1
3. **Branch divergence in DDTree**: Each branch may use different mask — but branches are sequential on CPU
4. **GPU batching**: Multiple substrates → different masks per batch element → gather/scatter overhead

---

## Relation to Commercial Strategy (Verdict 003)

### Engine/Fuel Split

| Layer | Prism Component | Our Mapping | License |
|-------|----------------|-------------|---------|
| Engine | SubstrateGate (dual sparsity + routing) | katgpt-rs `substrate_gate` feature | MIT (open) |
| Engine | SubstrateMask format + loader | katgpt-rs `SubstrateMask` type | MIT (open) |
| Engine | Recovery scoring trait | katgpt-rs `ScreeningPruner` extension | MIT (open) |
| **Fuel** | Collimated LoRA adapters | riir-ai training pipeline output | SaaS (private) |
| **Fuel** | Per-domain capability masks | riir-ai attribution → exported .mask files | SaaS (private) |
| **Fuel** | lora.bin (semantic accuracy) | Already in strategy | SaaS (private) |

The engine can load ANY mask (including open-source CNA-discovered masks). The moat is the **collimated masks** from riir-ai training — higher recovery, domain-specific, produced by the data flywheel.

### How It Serves the RIIR Wedge

For Python→Rust translation:
1. riir-ai trains collimated LoRA for "Python→Rust translation" capability
2. Attribution extracts a sparse mask (~5-20% of MLP)
3. katgpt-rs loads the mask at inference → translates at fraction of FLOPs
4. Multiple translation sub-capabilities (stdlib, async, error handling) each have their own mask
5. DDTree routes branches through appropriate substrates → better coverage
6. ConstraintPruner validates Rust syntax → guaranteed compilation

This is **"deploy one capability at fraction of cost"** — exactly Prism's conclusion applied to our commercial strategy.

---

## Research Gaps

1. **CNA vs ReLP mask quality** — CNA is forward-only, ReLP uses backward pass. Need to measure recovery difference on our models.
2. **Substrate mask transfer across model sizes** — Prism shows mask is model-specific (doesn't transfer). Need to verify for our domain.
3. **Multi-capability overlap** — When multiple capabilities share channels, can we serve them from one combined mask?
4. **GPU substrate batching** — Multiple different masks in one batch requires gather/scatter. Need benchmark.
5. **Recovery threshold for production** — What recovery % is acceptable for Python→Rust translation? 85%? 95%?

---

## TL;DR

Prism proves capabilities are sparse in MLP channel space. The modelless fusion is **SubstrateGate**: pre-computed capability masks (from CNA discovery) intersected with ReLU activation masks for dual sparsity at inference time. Combined with DDTree capability routing and ConstraintPruner validation, this enables capability-level speculative decoding at 10-40% FLOPs reduction. Feature-gated as `substrate_gate`, GOAT-gated with 7 criteria, default-on if all pass. The collimated LoRA adapters that produce higher-quality masks belong to riir-ai (fuel in the engine/fuel split).
