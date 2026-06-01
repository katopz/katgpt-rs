# Research 116: LLM Sleep — Offline Recursive Memory Consolidation

> **Paper:** [Language Models Need Sleep](https://arxiv.org/abs/2605.26099) — Lee, McLeish, Goldstein, Fanti (CMU/UMD), May 2026
> **Date:** 2026-05-27
> **Related Research:** 070 (GDN2), 073 (LT2), 069 (AutoDreamer), 097 (Training-Free Loop), 051 (Deep Manifold)
> **Related Plans:** 108 (LT2 Looped Inference), 105 (GDN2), 107 (AutoDreamer), 136 (Training-Free Loop)
> **Verdict: CONDITIONAL ACCEPT — Direct architectural fit with our GDN2+LT2 pipeline. Sleep-time consolidation is the missing piece between our wake-time LT2 looping and our modelless AutoDreamer consolidation. Feature-gate as `sleep_consolidation`. GOAT proof required before default-on. NOT a GOAT pillar — model infrastructure, not game product. Goes to katgpt-rs (model-based domain).**

---

## TL;DR

When a hybrid SSM-Attention model's KV cache fills up, instead of just evicting or compressing, perform **N offline recurrent passes ("sleep")** over the accumulated context to consolidate it into the SSM fast weights before clearing the KV cache. This preserves single-pass wake-time prediction latency while spending more compute on memory consolidation.

Key finding: Vanilla SSM-Attention hybrids fail on tasks requiring deep reasoning over evicted context, **even when memory capacity is sufficient**. The bottleneck is not storage but **computation available for transforming evicted context into useful internal state**. Increasing sleep passes N improves performance, especially on harder reasoning tasks.

Architecture: `Embed → [B_attn → B_ssm → ... → B_attn]×N → OutProj` — loop N times at eviction boundaries.

**For our stack:** We already have GDN2 (fast-weight memory) + LT2 (looped inference) + AutoDreamer (modelless consolidation). Sleep fills the gap: it's the **model-based analog of AutoDreamer's consolidation**, applied to our GDN2 fast weights instead of bandit arms. Our LT2 currently loops at wake time — sleep moves loops to eviction time, preserving wake-time single-pass latency for real-time game constraints.

---

## Core Mechanism

### The Problem: Reasoning Over Evicted Context

Hybrid SSM-Attention models (like our GDN2 + SDPA pipeline) handle long context by:
1. Storing recent tokens in KV cache (attention layers)
2. Compressing older tokens into fixed-size fast weights (SSM layers)

When KV cache fills → evict → old tokens only live in fast weights. Problem: converting context into useful fast weights is itself a nontrivial computation. A single forward pass may not be enough for deep reasoning.

**Experiment (cellular automaton Rule 110):** Model sees 4 binary strings, each 24 tokens. Must predict the state after t transitions. With t=0 (simple retrieval), hybrid model works. With t≥4 (deep reasoning), performance collapses — despite having enough fast-weight capacity to store all 4 states.

### Sleep: N Offline Recurrent Passes at Eviction Boundaries

```
Algorithm 1 (simplified):
1. Zero-initialize SSM fast weights S
2. Split input into chunks of size L (context window)
3. For each chunk:
   a. If consolidation phase (no loss):
      For n = 1..N:
        h, S ← Blocks(h, S)  // N recurrent passes
   b. If prediction phase:
      h, S ← Blocks(h, S)   // single pass only
      Loss ← CrossEntropy(output, target)
4. Backpropagate through entire process
```

**Key property:** During sleep, the model receives no external tokens. It recursively processes the **already-seen** context, iteratively refining the fast weights. After sleep, KV cache is cleared. Prediction phase is always single-pass.

### Fast-Weight Update View

The paper uses Gated Delta Networks (GDN) as the SSM block:

```
S_t = α_t · S_{t-1} + β_t · v_t · k_t^T    // standard GDN update
o_t = S_t · q_t                                // query from fast weights
```

During sleep, the same context is re-processed N times. Each pass applies a learned local update rule to S. The gradient flows through the refined fast weights (not the refined features, which are discarded after sleep).

This is conceptually similar to:
- **Iterative gradient descent** on the fast weights (but the update rule is learned, not a gradient step)
- **Hippocampal replay** in biological systems (reactivate memories during sleep to consolidate into cortex)

---

## Key Results

### Synthetic Task: Cellular Automaton (Rule 110)

| Model | t=0 (retrieval) | t=4 | t=8 | t=14 | t=32 |
|-------|----------------|-----|-----|------|------|
| Hybrid (no loop) | 100% | ~75% | ~50% | ~25% | ~10% |
| Hybrid + 2 loops | 100% | ~85% | ~65% | ~45% | ~20% |
| Hybrid + 4 loops | 100% | ~95% | ~85% | ~75% | ~35% |

As reasoning depth (t) increases, sleep duration (N) becomes increasingly important.

### Synthetic Task: Depo (Multi-Hop Graph Retrieval)

| Hops | No Loop | 2 Loops | 4 Loops |
|------|---------|---------|---------|
| 1-hop | Low loss | Low loss | Low loss |
| 2-hop | Low loss | Low loss | Low loss |
| 4-hop | Stalled | Learning | Learning |
| 8-hop | Stalled | Stalled | Learning |
| 16-hop | Stalled | Stalled | Beginning |

Only 4-loop model makes progress on 16-hop queries. Each additional loop unlocks deeper reasoning.

### Realistic Task: GSM-Infinite (Math Reasoning)

**Jet-Nemotron 2B (SSM-Attention hybrid):**

| Operations | No Loop | 6 Loops | Improvement |
|------------|---------|---------|-------------|
| 2-op | 0.985 | 0.979 | -0.6% (saturated) |
| 4-op | 0.995 | 0.994 | -0.1% (saturated) |
| 6-op | 0.742 | **0.812** | **+9.4%** |
| 8-op | 0.351 | **0.388** | **+10.5%** |

**Ouro 1.4B (looped attention-only):**

| Operations | No Loop | 4 Loops | Improvement |
|------------|---------|---------|-------------|
| 2-op | 0.857 | **0.868** | +1.3% |
| 4-op | 0.903 | **0.932** | +3.2% |
| 6-op | 0.419 | **0.615** | **+46.8%** |
| 8-op | 0.209 | **0.272** | **+30.1%** |

**Key insight:** Gains are largest on hardest problems. Easy problems are saturated regardless.

### Sliding-Window Eviction (L=512, Ouro 1.4B)

| Operations | No Loop | 4 Loops | Improvement |
|------------|---------|---------|-------------|
| 2-op | 0.596 | **0.905** | **+51.8%** |
| 8-op | 0.116 | **0.137** | +18.1% |

When active attention window is much smaller than sequence length, sleep helps both retrieval AND reasoning.

---

## Training Throughput

Sleep makes training sequential across context windows (can't process window j+1 until window j is done). However:

1. **Window-axis parallelism loss is minimal** when window size L is large enough to saturate GPU
2. **Recurrent-depth cost grows linearly** with N (2 loops ≈ 50% throughput, 4 loops ≈ 25% throughput)
3. **Activation checkpointing** across context chunks prevents OOM

---

## Mapping to Our Architecture

### What We Already Have

| Sleep Component | Our Equivalent | Status |
|-----------------|---------------|--------|
| GDN SSM block | GDN2 (Plan 105) | ✅ Implemented, 14/14 GOAT |
| Loop weight sharing | LT2 (Plan 108) | ✅ Implemented, 11/11 GOAT |
| Fast-weight state (S) | GDN2 recurrent state | ✅ In `gdn2_recurrent_step` |
| KV cache eviction | TurboQuant/SpectralQuant | ✅ Compression, not eviction |
| Offline consolidation (modelless) | AutoDreamer (Plan 107) | ✅ For bandit/δ-mem, not model weights |
| Context→weights | Freeze/Thaw (Plan 092) | ✅ But single-pass, no recurrence |
| Training-free loop | Plan 136 | ✅ But wake-time, not sleep-time |

### The Gap: Sleep-Time Consolidation for Model Weights

| What We Have | What Sleep Adds |
|-------------|-----------------|
| LT2 loops at **wake time** (inference) | Sleep loops at **eviction time** (consolidation) |
| AutoDreamer consolidates **bandit arms** | Sleep consolidates **GDN2 fast weights** |
| Freeze/Thaw is **single-pass** | Sleep is **N-pass recurrent** |
| KV cache is **compressed** (TurboQuant) | KV cache is **evicted** after consolidation |
| Training-free loop for **same weights** | Sleep **refines** fast weights |

**Sleep is the missing link between our modelless AutoDreamer and our wake-time LT2.**

### Where Sleep Fits

```
Game Session (long context, 2000+ tokens)
  │
  ├── [Context Window fills at L tokens]
  │
  ├── OPTION A (current): Compress KV cache with TurboQuant/SpectralQuant
  │   └── Pro: No extra compute. Con: Lossy compression, quality degrades at long context.
  │
  ├── OPTION B (LT2 wake-time): Loop T times during inference
  │   └── Pro: Better quality. Con: Increases wake-time latency (bad for 20Hz game loop).
  │
  └── OPTION C (Sleep-time): Loop N times at eviction, then clear KV cache
      └── Pro: Best quality at eviction boundary, single-pass wake time.
          Con: Training requires BPTT through N passes, sequential across windows.
```

**For our real-time game constraint (Pillar 4: 20Hz frame sampling), OPTION C is ideal:**
- Spend compute during eviction (offline, not latency-sensitive)
- Keep wake-time single-pass (20Hz budget preserved)
- Better quality than compression (N recurrent refinement passes)

### Game-Specific Benefits

| Pillar | How Sleep Helps | Mechanism |
|--------|----------------|-----------|
| Pillar 3 (NPC Dialog) | Long NPC conversations compressed into compact fast weights | Sleep consolidation of dialog context → NPC remembers full conversation history |
| Pillar 4 (Frame-Sampling) | Long combat sessions processed without wake-time latency | Eviction-time consolidation keeps 20Hz budget intact |
| Freeze/Thaw | Better context→weights transfer | N-pass recurrent refinement > single-pass freezing |
| LoRA Training | Sleep as pre-training for LoRA adapters | Consolidated fast weights provide better initialization |

**But:** Sleep is model-based (requires SSM blocks), not modelless. The mechanism is generic architecture, not game-specific. This goes to **katgpt-rs** (model infrastructure).

---

## Relationship to Existing Research

### vs. AutoDreamer (Research 069)

| Aspect | AutoDreamer | LLM Sleep |
|--------|-------------|-----------|
| Domain | Modelless (bandit arms, Q-values) | Model-based (SSM fast weights) |
| Consolidation | Deterministic merge + counterfactual | Learned recurrent update rule |
| Target | Game strategy memory | Model internal state |
| Compute | O(|region|) deterministic | O(N × L × d²) learned |
| Training | No gradients needed | BPTT through N passes |

**Complementary, not competing.** AutoDreamer consolidates our modelless memory; Sleep consolidates our model-based memory. Both use the same two-timescale principle (fast online → slow offline).

### vs. LT2 (Research 073)

| Aspect | LT2 | LLM Sleep |
|--------|-----|-----------|
| When to loop | Wake time (inference) | Sleep time (eviction) |
| What's looped | All layers | All layers (same) |
| Wake-time cost | T× increase | No increase (single-pass) |
| Sleep-time cost | None | N× increase |
| Training | Optional (training-free wrapper exists) | Required (BPTT through sleep) |

**Sleep moves compute from wake-time to sleep-time.** For our real-time game loop, this is strictly better — we want minimal wake-time latency.

### vs. Freeze/Thaw (Plan 092)

Freeze/Thaw is single-pass context→weights. Sleep is N-pass recurrent. Sleep could replace or augment the freeze step with multiple consolidation passes before thawing.

### vs. Deep Manifold Fixed Points (Research 051)

Sleep iteratively refines fast weights toward stable representations. This is conceptually related to finding fixed points in the fast-weight update dynamics. The N recurrent passes can be viewed as N steps of a fixed-point iteration. However, sleep's update rule is learned (not a gradient step), which is more flexible but requires training.

---

## Architecture for Our Stack

### Feature Gate: `sleep_consolidation`

```toml
[features]
sleep_consolidation = ["lt2_looped", "gdn2_attention"]
# Requires: LT2 loop infrastructure + GDN2 fast-weight blocks
# Optional dependency on: AutoDreamer (for hybrid modelless+model-based consolidation)
```

### New Types

```rust
/// Sleep consolidation configuration.
/// Applied at eviction boundaries before clearing KV cache.
#[derive(Clone, Debug)]
pub struct SleepConfig {
    /// Number of recurrent passes during sleep (paper default: 2-6).
    pub sleep_passes: usize,
    /// Eviction strategy: hard (clear all) or sliding-window (keep last L-1).
    pub eviction: EvictionStrategy,
    /// Context window size L (tokens per chunk).
    pub window_size: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Clear entire KV cache after sleep.
    #[default]
    HardEvict,
    /// Keep most recent L-1 tokens, evict older.
    SlidingWindow,
}
```

### Integration Point

The sleep mechanism integrates into our existing LT2 pipeline at the eviction boundary:

```rust
// In the LT2 looped forward pass (Plan 108), add sleep at eviction:
if is_at_eviction_boundary(position, config.window_size) {
    // SLEEP PHASE: N recurrent passes over current context
    for n in 0..config.sleep_passes {
        for layer in 0..config.n_layer {
            // GDN2 layers: update fast weights S
            // Attention layers: re-process current KV cache
            forward_layer(layer, &mut hidden, &mut fast_weights, &mut kv_cache);
        }
    }
    // EVICTION: clear KV cache (fast weights already consolidated)
    kv_cache.clear();
}
// WAKE PHASE: single-pass prediction (standard LT2)
forward_single_pass(&mut hidden, &mut fast_weights, &kv_cache);
```

### Module Structure

```text
src/sleep/
├── mod.rs              # Index, re-exports
├── types.rs            # SleepConfig, EvictionStrategy
├── consolidation.rs    # N-pass recurrent consolidation loop
├── eviction.rs         # Hard/sliding-window eviction after sleep
└── training.rs         # BPTT through sleep (requires gradients)
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training instability (BPTT through N passes) | High | Start with N=2, use gradient checkpointing. Paper confirms GDN is most stable mixer. |
| Sequential training across windows | Medium | GPU still saturated if L ≥ 1024. Our game contexts are 500-2000 tokens. |
| No training infrastructure yet | High | Training in riir-ai. Inference-only sleep (Plan 136 wrapper) is untrained baseline. |
| Sleep overhead at eviction | Low | Happens offline, not on the 20Hz path. |
| Fast-weight drift across sessions | Medium | Freeze/Thaw provides persistent state. Sleep adds transient consolidation. |
| Over-consolidation (lose details) | Low | Paper shows gains increase with N. Our AutoDreamer counterfactual utility can detect quality regression. |

---

## Open Questions

1. **Sleep with our GDN2 vs paper's GDN?** Our GDN2 has channel-wise erase/write (decoupled from the paper's scalar-gated GDN). The channel-wise gating should make consolidation more expressive — each channel can independently retain or forget.

2. **Optimal N for CPU inference?** Paper uses N=2-6 on GPU. On CPU, each pass is sequential. N=2 may be the sweet spot for our game context sizes.

3. **Sleep + TurboQuant hybrid?** Instead of evicting the full KV cache, we could: sleep → consolidate → compress remaining KV cache with TurboQuant. This gives both recurrent consolidation AND compressed retention.

4. **Sleep for modelless path?** The paper requires SSM blocks (model-based). For our modelless path (bandit/δ-mem), AutoDreamer already provides consolidation. The question is: can sleep's N-pass recurrent refinement principle be applied to modelless consolidation (e.g., N passes of arm merging)?

5. **Pre-trained model initialization?** The paper fine-tunes pre-trained models (Jet-Nemotron, Ouro). We'd need to either (a) adapt from our existing models, or (b) train from scratch. Option (a) is more practical.

---

## Decision Matrix Assessment

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT passed | ❌ | Not implemented. Requires GOAT proof before promotion. |
| MMO-product | ⬜ | Indirect — better long-context reasoning for game sessions |
| LoRA-independent | ✅ | Updates SSM fast weights, not LoRA. Works with or without LoRA. |
| Defensible | ⬜ | Paper is public. Application to real-time game eviction is somewhat novel. |
| Secret coverage | ⬜ | Indirect — improves model quality, doesn't directly protect secrets |

**Classification:** Infrastructure improvement, not a pillar. Strengthens existing pillars (3, 4) if it works. Fails gracefully if it doesn't (TurboQuant/SpectralQuant still work).

---

## Proposed Plan (katgpt-rs Plan 154)

### Tasks

- [x] T1: Add `sleep_consolidation` feature gate to `Cargo.toml` (depends on `lt2_looped`, `gdn2_attention`) — `Cargo.toml` L107-108, in default features
- [x] T2: Implement `SleepConfig`, `EvictionStrategy` types in `src/sleep/types.rs` — `SleepConfig` + `EvictionStrategy { HardEvict, SlidingWindow { retain } }`
- [x] T3: Implement `consolidation_pass()` — single recurrent pass through all layers with fast-weight carry — `src/sleep/consolidation.rs` L39
- [x] T4: Implement `sleep()` — N calls to `consolidation_pass()` at eviction boundary — `src/sleep/consolidation.rs` L116
- [x] T5: Implement `eviction::HardEvict` — clear full KV cache after sleep — `src/sleep/eviction.rs::hard_evict()`, `#[default]`
- [x] T6: Implement `eviction::SlidingWindow` — retain last L-1 tokens after sleep — `src/sleep/eviction.rs::sliding_window_evict()`
- [x] T7: Integrate sleep into LT2 forward pass (eviction boundary hook) — `forward_looped()` in `transformer.rs` wired with `Option<&mut MultiLayerGdn2Cache>` + `Option<&SleepConfig>`, hook after LM head
- [x] T8: GOAT proof — sleep vs no-sleep on synthetic reasoning task (Depo-style multi-hop) — `tests/bench_154_sleep_consolidation_goat.rs::goat_t10_sleep_vs_nosleep_multihop()`
- [x] T9: GOAT proof — sleep + TurboQuant hybrid vs TurboQuant-only on long-context task — `tests/bench_154_sleep_consolidation_goat.rs::goat_t11_sleep_with_quantized_context()`
- [x] T10: GOAT proof — sleep on game context (long Bomber session, long NPC dialog) — `tests/bench_154_sleep_consolidation_goat.rs::goat_t12_game_context_long_session()`
- [x] T11: Benchmark — sleep overhead (N=2,4,6) vs no-sleep vs LT2-wake-time — `tests/bench_154_sleep_consolidation_goat.rs::goat_t13_sleep_overhead_benchmark()`
- [x] T12: Update README + .docs — `.docs/18_sleep_consolidation.md`

### Priority: MEDIUM

Not blocking any pillar. LT2 wake-time looping works today. Sleep is an optimization for when we have training infrastructure (riir-ai). Implement inference-only first, add training later.

---

## References

- LLM Sleep paper: https://arxiv.org/abs/2605.26099
- Gated DeltaNet (GDN): https://arxiv.org/abs/2412.06464
- Gated DeltaNet-2 (our distillation): Research 070, Plan 105
- LT2 Looped Transformers: https://arxiv.org/abs/2605.20670
- LT2 (our implementation): Research 073, Plan 108
- AutoDreamer (our modelless consolidation): Research 069, Plan 107
- GSM-Infinite benchmark: https://arxiv.org/abs/2505.20246
- Freeze/Thaw pipeline: Plan 092
