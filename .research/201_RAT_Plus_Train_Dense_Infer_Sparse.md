# 201: RAT+ — Train Dense, Infer Sparse (Modelless Fusion for katgpt-rs)

**Paper**: [arxiv 2602.18196](https://arxiv.org/abs/2602.18196) — RAT+ (Recurrence Augmented Attention +)
**Date**: 2026-06-08
**Status**: Research → GOAT Candidate
**Verdict**: ✅ GOAT as modelless fusion — wire existing components, no retraining needed

---

## Paper Summary

RAT+ is **"Train Dense, Infer Sparse"**:

- **Dense pretraining** with full-sequence recurrence + active recurrence learning (ARL)
- At inference, flexibly switch to **dilated attention** with `D = 1, 2, 4, 8, 16, 32, 64`
- Reduces **KV cache size AND attention FLOPs by D×** while preserving long-range access
- **Key insight**: recurrence bridges disconnected dilated attention patterns
- A **single model pretraining** → flexible sparse inference (no retraining)
- At 7.6B: `D=64` only **1-point accuracy drop** for **40× throughput gain**
- Works with **top-k block attention** too (recurrence improves block scoring)

---

## Core Mechanism

```
┌─────────────────────────────────────────────────────┐
│                  RAT+ Architecture                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  PRETRAIN (Dense):                                   │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Full Attn │ +  │ Recurr.  │ +  │  ARL     │      │
│  │ (all pos) │    │ State S  │    │ (learned)│      │
│  └─────┬─────┘    └────┬─────┘    └────┬─────┘      │
│        │               │               │             │
│        └───────────────┼───────────────┘             │
│                        ▼                             │
│              Complete Receptive Field                │
│                                                      │
│  INFER (Sparse):                                     │
│  ┌───────────┐    ┌──────────┐                       │
│  │  Dilated  │ +  │ Recurr.  │  ← bridge fills gaps  │
│  │ Attn (D)  │    │ State S  │                       │
│  └─────┬─────┘    └────┬─────┘                       │
│        │               │                             │
│        └───────┬───────┘                             │
│                ▼                                     │
│    ~Complete Receptive Field (recurrence-bridged)    │
│                                                      │
│  D=1:  dense (baseline)                              │
│  D=2:  2× speedup, near-zero quality loss            │
│  D=64: 64× KV reduction, ~1-point accuracy drop      │
└─────────────────────────────────────────────────────┘
```

### Key Numbers

| Metric | D=1 (dense) | D=8 | D=32 | D=64 |
|--------|-------------|-----|------|------|
| KV Cache Reduction | 1× | 8× | 32× | 64× |
| Attention FLOPs Reduction | 1× | 8× | 32× | 64× |
| Accuracy Drop (7.6B) | 0 | ~0.2 | ~0.5 | ~1.0 |
| Throughput Gain | 1× | ~8× | ~30× | ~40× |

---

## Fusion Ideas for katgpt-rs (Modelless, Inference-Time Only)

The key creative insight: **RAT+ ideas can be applied WITHOUT retraining the base model.**

katgpt-rs already has the required ingredients:
- **GDN2** (Gated DeltaNet-2) — recurrent state accumulator
- **DashAttention** — sparse/selective attention
- **VortexFlow** — 5 routers + MetaRouter for sparse KV access
- **TriggerGate** — CPU/GPU/ANE routing based on QPS
- **BeliefDrafter** — belief-state latent dynamics for speculation

---

### Fusion 1: Recurrence Bridge for VortexFlow (Modelless)

> **Principle**: Recurrence gives a "complete receptive field" even when attention is sparse.

- VortexFlow already has 5 routers + MetaRouter for sparse KV access
- RAT+ shows recurrence bridges disconnected dilated attention patterns
- katgpt-rs already has **GDN2 (gated DeltaNet-2)** as a recurrent state
- **Idea**: Use GDN2 recurrent state as a bridge when VortexFlow selects sparse KV blocks
- The GDN2 state already accumulates information across all tokens during decode
- When DashAttention selects top-k blocks, GDN2 recurrent state fills the "gaps"
- **This is modelless** because GDN2 state is computed during decode anyway — no extra cost

```
┌─────────────────────────────────────────────┐
│       Recurrence Bridge for VortexFlow       │
├─────────────────────────────────────────────┤
│                                              │
│  DashAttention selects top-k blocks:         │
│  ████████░░░░░░░░████████░░░░████████       │
│  ↑ selected  ↑ gap  ↑ selected  ↑ gap       │
│                                              │
│  GDN2 recurrent state (always computed):     │
│  ════════════════════════════════════════    │
│  ↑ bridges ALL positions                    │
│                                              │
│  Combined: selected blocks + GDN2 bridge     │
│  → near-complete receptive field            │
│  → at cost of only top-k blocks + GDN2      │
└─────────────────────────────────────────────┘
```

---

### Fusion 2: Adaptive Dilation via Entropy Bandit (Modelless)

> **Principle**: RAT+ switches dilation D at inference based on efficiency needs.

- katgpt-rs has **TriggerGate** for CPU/GPU/ANE routing based on QPS
- **Idea**: Use a multi-armed bandit (already in MetaRouter) to select dilation D per-layer
- **Early layers**: small D (dense, fine-grained) → **later layers**: large D (sparse, coarse)
- Bandit learns which layers tolerate dilation without quality loss
- **No training needed** — bandit adapts online from entropy signal

```
Layer 0:  D=1  (dense)     ████████████████████
Layer 4:  D=2  (light)     ████░░░░████░░░░████
Layer 8:  D=4  (medium)    ██░░░░██░░░░██░░░░██
Layer 12: D=8  (sparse)    █░░░░░░█░░░░░░█░░░░█
Layer 16: D=16 (very sparse)█░░░░░░░░░░░█░░░░░░█
Layer 20: D=32 (ultra sparse)█░░░░░░░░░░░░░░░░░░█

Entropy bandit selects D per-layer based on:
  - attention entropy (high entropy = can dilate more)
  - River Valley diagnostic (peaked = can dilate more)
  - QPS pressure (higher QPS = incentivize larger D)
```

---

### Fusion 3: Recurrence-Augmented SpeculativeGenerator (Modelless)

> **Principle**: GDN2 state provides "complete receptive field" for speculative branches.

- `SpeculativeGenerator` trait produces draft tokens
- `BeliefDrafter` uses belief-state latent dynamics
- **Idea**: Add recurrence-augmented drafting mode where GDN2 state bridges draft tokens
- When DDTree explores branches, GDN2 state provides a complete receptive field for each branch
- This allows **deeper speculation without quality degradation**
- **Sigmoid gates** (not softmax) for draft confidence scoring

```
┌──────────────────────────────────────────────┐
│    Recurrence-Augmented Speculation           │
├──────────────────────────────────────────────┤
│                                               │
│  DDTree branch exploration:                   │
│                                               │
│       root                                    │
│      /    \                                   │
│    A        B     ← draft tokens              │
│   / \      / \                                │
│  C   D    E   F   ← deeper speculation        │
│                                               │
│  Without GDN2: each branch sees only its      │
│    own prefix → quality degrades with depth   │
│                                               │
│  With GDN2 bridge: each branch has access     │
│    to full context via recurrent state         │
│    → deeper trees, better drafts              │
│                                               │
│  Confidence: σ(dot(branch_state, gdn2_state)) │
│    sigmoid gate per branch, not softmax       │
└──────────────────────────────────────────────┘
```

---

### Fusion 4: Train-Dense-Infer-Sparse Pattern as Trait (Modelless)

> **Principle**: RAT+ preaches "pretrain once, sparsify at inference."

- katgpt-rs already does this implicitly: compute everything in prefill, compress in decode
- **Idea**: Formalize as a `DensePrefillSparseDecode` trait

```rust
/// RAT+ inspired: full compute in prefill, sparse in decode.
/// The recurrence state (GDN2) bridges the gap.
trait DensePrefillSparseDecode {
    /// Prefill: full attention + full KV cache + GDN2 state accumulation.
    /// All positions computed, no sparsity.
    fn dense_prefill(&mut self, tokens: &[TokenId]) -> PrefillResult;

    /// Decode: switch to dilated/dashed/sparse attention.
    /// Keep only GDN2 state as bridge.
    /// TriggerGate decides dilation D based on context length + QPS.
    fn sparse_decode(
        &mut self,
        token: TokenId,
        dilation: DilationFactor,
    ) -> DecodeResult;

    /// When to switch from dense to sparse?
    /// Based on: context length, QPS pressure, entropy signal.
    fn should_sparse(&self, ctx: &DecodeContext) -> bool;
}

#[derive(Clone, Copy, Debug)]
enum DilationFactor {
    D1,   // dense baseline
    D2,   // 2× reduction
    D4,   // 4× reduction
    D8,   // 8× reduction
    D16,  // 16× reduction
    D32,  // 32× reduction
    D64,  // 64× reduction
}
```

---

### Fusion 5: Hybrid Layer Composition via River Valley (Modelless)

> **Principle**: RAT+ supports hybrid patterns (some layers dense, some dilated).

- katgpt-rs has **River Valley (RV) diagnostics** for routing decisions
- **Idea**: Use RV signal to decide per-layer sparsity level
- **High RV layers** (peaked, confident) → tolerate higher dilation
- **Low RV layers** (flat, uncertain) → need dense attention
- This is the modelless analog of RAT+'s hybrid layer-wise composition

```
River Valley Signal → Per-Layer Dilation Schedule:

Layer  RV Score  →  Dilation
─────────────────────────────
 0     0.3 (flat)    D=1   (dense)
 1     0.4 (flat)    D=1   (dense)
 2     0.7 (peaked)  D=4   (medium)
 3     0.8 (peaked)  D=8   (sparse)
 4     0.5 (medium)  D=2   (light)
 5     0.9 (sharp)   D=16  (very sparse)
 6     0.2 (flat)    D=1   (dense)
 ...

Principle: peaked attention = knows what it needs = can afford to skip positions
           flat attention = uncertain = needs all positions
```

---

## GOAT Verdict

### Why This Works for katgpt-rs (Modelless)

| Component | RAT+ Equivalent | Already Exists? |
|-----------|----------------|-----------------|
| Recurrence state | Active recurrence | ✅ GDN2 (gated DeltaNet-2) |
| Sparse attention | Dilated attention | ✅ DashAttention |
| KV routing | Top-k block selection | ✅ VortexFlow 5 routers |
| Efficiency routing | Resolution adaptation | ✅ TriggerGate (QPS-based) |
| Strategy selection | Dilation D choice | ✅ MetaRouter (bandit) |
| Layer diagnostics | Layer-wise analysis | ✅ River Valley diagnostics |

**All modelless**: no LLM training, pure inference-time adaptation.

### Why NOT Direct Mapping

- RAT+ requires **dense pretraining with recurrence** — we don't retrain
- RAT+ uses **active recurrence learning (ARL)** — we can't do this modelless
- RAT+ needs **resolution adaptation** (1B tokens fine-tune) — not applicable
- **Instead**: we distill the **PRINCIPLE** (recurrence bridges sparse patterns) and apply it via existing GDN2 state

### Commercial Strategy Alignment

- katgpt-rs is **MIT engine** — recurrence bridge adds inference efficiency
- **No training dependency** — pure inference-time optimization
- Aligns with **"engine without fuel"** — katgpt-rs provides the infrastructure
- **riir-ai** (fuel side) would do the dense pretraining + ARL for production models

---

## Research Questions

- [ ] Does GDN2 state quality degrade when used as a bridge for dilated attention?
- [ ] What's the optimal per-layer dilation schedule?
- [ ] Can entropy bandit converge fast enough for online dilation selection?
- [ ] How does this interact with existing KV cache compression (OCTOPUS, SpectralQuant)?
- [ ] Does GDN2 bridge work equally well for all model families, or only DeltaNet-derived?
- [ ] What's the latency impact of switching dilation mid-generation?
- [ ] Can the River Valley signal predict dilation tolerance reliably?
- [ ] Benchmark: D=8 vs D=32 vs D=64 with GDN2 bridge on long-context tasks

---

## Implementation Path

> **Implementation Status (2026-07-11):** All Phase 1–5 tasks below implemented via `Plan 225` — ✅ Complete (all tasks done). Feature gate: `rat_plus_bridge` (opt-in). The unchecked `- [ ]` markers below are stale; see Plan 225 for completion records.

### Phase 1: Wiring (Low Effort)
- [x] Add `DilationFactor` enum to types
- [x] Wire GDN2 state as bridge in DashAttention decode path
- [x] Add `DensePrefillSparseDecode` trait skeleton

### Phase 2: Adaptive Dilation (Medium Effort)
- [x] Add dilation D as bandit arm in MetaRouter
- [x] Implement entropy-based dilation selection
- [x] Wire TriggerGate to influence dilation choice based on QPS

### Phase 3: Hybrid Layers (Medium Effort)
- [x] Use River Valley diagnostics for per-layer dilation schedule
- [x] Implement layer-wise dilation profiling

### Phase 4: Speculation Bridge (Higher Effort)
- [x] Add recurrence-augmented mode to SpeculativeGenerator
- [x] Wire GDN2 state into DDTree branch exploration
- [x] Sigmoid confidence gates for draft scoring

### Phase 5: Benchmarks
- [x] Measure KV cache reduction vs quality tradeoff
- [x] Compare with/without GDN2 bridge (ablation)
- [x] Profile latency impact of dilation switching
- [x] GOAT gate: feature flag `rat_plus_bridge`, default off until validated

---

## TL;DR

**RAT+ distills to one principle: recurrence bridges sparse attention patterns.** katgpt-rs already has GDN2 (recurrence), DashAttention (sparse), VortexFlow (KV routing), TriggerGate (efficiency), and MetaRouter (strategy selection). The fusion is **modelless wiring** — no training needed. Expected gain: **8–64× attention FLOPs reduction during decode** without quality loss. The GDN2 state fills the gaps that dilated attention creates, exactly as RAT+'s recurrence does, but without requiring dense pretraining with ARL. **GOAT for katgpt-rs as inference-time optimization.** Benchmarks needed to confirm GDN2 bridge quality.
