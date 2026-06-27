# katgpt-rs: Sleep Consolidation — Offline Recursive Memory Consolidation at Eviction

> **Plan 154** · **Feature gate:** `sleep_consolidation` (default-ON, requires `lt2_looped` + `gdn2_attention`)
> **Reference:** arXiv:2605.26099 — Lee et al., May 2026

## 1. Overview

Sleep consolidation moves LT2's wake-time looping to **eviction-time consolidation**. When the KV cache fills, N offline recurrent passes bake the cached context into GDN2 fast-weight state before evicting the cache. This preserves single-pass wake-time latency for real-time game constraints (≤50ms at 20Hz).

**Key insight:** Sleep is the model-based analog of AutoDreamer (Plan 107), applied to GDN2 fast weights instead of modelless logits.

### Core Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `sleep_passes` | 2 | Number of recurrent consolidation passes at eviction boundary |
| `window_size` | 512 | KV cache capacity threshold that triggers sleep |
| `eviction` | `HardEvict` | Strategy for clearing KV cache after consolidation |

---

## 2. Architecture

```
Existing LT2 Pipeline:
  Input → [SDPA → GDN2 → SDPA → GDN2 → ...]×T (wake-time loops) → Output

With Sleep:
  Input → Context fills → [SDPA → GDN2 → ...]×N (sleep-time consolidation) → Evict KV → Continue
         ↑ Single-pass at wake time (T=1)                    ↑ N-pass at eviction boundary
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sleep Pipeline                              │
│                  (src/sleep/consolidation.rs)                   │
│                                                                 │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────────┐  │
│  │ KV Cache │───▶│ consolidation_   │───▶│ GDN2 Fast-Weight  │  │
│  │  (full)  │    │ pass() × N       │    │ State S (updated) │  │
│  └──────────┘    └──────────────────┘    └───────────────────┘  │
│                          │                        │             │
│                          ▼                        ▼             │
│                  ┌──────────────┐        Context now in         │
│                  │  evict()     │        recurrent state        │
│                  │ HardEvict or │        (O(1) decode)          │
│                  │ SlidingWindow│                               │
│                  └──────┬───────┘                               │
│                         ▼                                       │
│                  KV cache cleared                               │
│                  → continue generation                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Structure

```text
src/sleep/
├── mod.rs              # Index, re-exports: sleep, consolidation_pass, SleepConfig, EvictionStrategy
├── types.rs            # SleepConfig, EvictionStrategy enum
├── consolidation.rs    # N-pass recurrent consolidation loop + sleep() entry point
└── eviction.rs         # HardEvict / SlidingWindow eviction after consolidation
```

---

## 4. API

### `SleepConfig`

```rust
let config = SleepConfig {
    sleep_passes: 4,                           // 4 consolidation passes
    eviction: EvictionStrategy::SlidingWindow { retain: 8 },
    window_size: 1024,
};

// Check if sleep should trigger
if config.should_sleep(current_pos) {
    sleep(&mut ctx, &weights, &mut kv_cache, &mut gdn2_cache, &config, &model_config);
}
```

### `SleepConfig` Constructors

```rust
let config = SleepConfig::new(4);                    // 4 passes, default eviction + window
let config = SleepConfig::with_sliding_window(4, 8); // 4 passes, SlidingWindow { retain: 8 }
```

### `consolidation_pass(kv_cache, gdn2_cache, fill_pos, config)`

Single recurrent pass: replays all cached K/V pairs through `gdn2_recurrent_step()`, updating fast-weight state S in-place. Uses K as both key and query (self-consolidation) with L2 normalization via `gdn2::kernel::l2_normalize`.

### `sleep(ctx, weights, kv_cache, gdn2_cache, sleep_config, config) → usize`

Main entry point: N× `consolidation_pass()` + `evict()`. Returns the number of passes performed (0 if cache was empty). Note: `ctx` (`ForwardContext`) and `weights` (`TransformerWeights`) are currently unused but reserved for future layer-norm passes.

### `EvictionStrategy`

| Variant | Behavior |
|---------|----------|
| `HardEvict` | Zeros entire KV cache + resets fill_pos to 0 |
| `SlidingWindow { retain }` | Shifts last `retain` tokens to front, zeros the rest |

---

## 5. Integration Points

| Component | Change | Gate |
|-----------|--------|------|
| `Cargo.toml` | `sleep_consolidation = ["lt2_looped", "gdn2_attention"]` (default-ON) | Feature |
| `src/lib.rs` | `pub mod sleep;` | `#[cfg(feature = "sleep_consolidation")]` |
| `gdn2::kernel::gdn2_recurrent_step` | Core consolidation primitive (already exists) | `gdn2_attention` |
| `gdn2::kernel::l2_normalize` | L2 normalization for self-consolidation stability | `gdn2_attention` |
| `gdn2::types::MultiLayerGdn2Cache` | GDN2 fast-weight state per layer/head with scratch buffers | `gdn2_attention` |
| `transformer::MultiLayerKVCache` | KV cache with fill_pos(), reset(), advance_pos() | `lt2_looped` |
| `transformer::ForwardContext` | Forward context (reserved, currently unused) | `lt2_looped` |
| `transformer::TransformerWeights` | Weights (reserved, currently unused) | `lt2_looped` |

---

## 6. GOAT Proof Criteria

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Multi-hop accuracy | ≥15% improvement over no-sleep at 8-hop | Paper shows 30-47% on hardest tasks |
| Long-context quality | ≥5% improvement at 4× window length | Paper shows 9-10% on GSM-Infinite 6-op |
| Wake-time latency | ≤5% increase over single-pass | Sleep is offline; wake stays single-pass |
| Game context | ≥10% improvement on >2000-token game session | Game-specific validation |

---

## 7. Testing

```bash
# Run all sleep module tests
cargo test --features sleep_consolidation --lib -- sleep::

# Full feature check
cargo check --features full
```

12 unit tests cover: `SleepConfig` defaults, boundary conditions, and constructors (`new`, `with_sliding_window`), `EvictionStrategy` variants (hard evict zeros all, sliding window retains recent, noop when under retain), `consolidation_pass` state updates and finiteness, `sleep()` with hard/empty caches, multi-pass strengthening. Distributed across `types.rs` (4 tests), `consolidation.rs` (5 tests), `eviction.rs` (3 tests).

---

## 8. References

- **Paper:** [arXiv:2605.26099](https://arxiv.org/abs/2605.26099) — LLM Sleep: Offline Recursive Memory Consolidation
- **Research 116:** Detailed distillation and analysis
- **Plan 108 (LT2):** Looped inference pipeline — weight-shared T-pass loop
- **Plan 105 (GDN2):** Gated DeltaNet-2 recurrent attention — O(1) decode
- **Plan 107 (AutoDreamer):** Modelless consolidation complement
- **Plan 092 (Freeze/Thaw):** Context→weights pipeline

---

## 9. Relationship to Sleep-Time Query Anticipation (Plan 334)

Plan 154 (this doc) and Plan 334 are **complementary sleep-time approaches**, both offline, both modelless, both serving the same goal — move compute off the wake-time critical path — but they differ in *where the consolidated state lives*:

| Aspect | Plan 154 (Sleep Consolidation) | Plan 334 (Sleep-Time Query Anticipator) |
|--------|--------------------------------|------------------------------------------|
| **Paper** | [arXiv:2605.26099](https://arxiv.org/abs/2605.26099) (LLM Sleep, Lee et al.) | [arXiv:2504.13171](https://arxiv.org/abs/2504.13171) (Sleep-time Compute, Lin et al.) |
| **Approach** | **State internalization** — bake context into recurrent fast-weight state via N-pass consolidation at the KV-eviction boundary. | **Artifact emission** — pre-compute anticipated-query answers into a reusable BLAKE3-committed `AnticipatedQuerySet` (the c' artifact). |
| **Where the answer lives** | Implicit in the recurrent state S (must be unrolled at wake-time to decode). | Explicit in c'.slots[i].precomputed (direct latent-vector lookup at wake-time). |
| **Wake-time cost** | Single recurrent step (O(1) decode, but needs the recurrent kernel). | Dot-product + sigmoid gate + linear blend (zero-alloc, ~57ns at D=64). |
| **Query-aware?** | No — consolidation is unconditional on what will be asked. | Yes — predictability scoring decides *which* queries are worth pre-computing. |
| **Economics** | Implicit (the cost is the consolidation passes; the benefit is preserved wake-time latency). | Explicit via `AmortizationCostModel` — `should_pre_compute(sleep_cost, N, E[gate])` answers whether pre-computing pays off for this context given N consumers. |
| **Failure mode** | Catastrophic forgetting if the recurrent state can't hold everything. | Cache miss on unpredictable queries → fall through to `fresh_think` (graceful). |

**Composability:** the two are not mutually exclusive. A consumer could use Plan 154 to consolidate long-range context into recurrent state (so wake-time decode stays O(1)), *and* use Plan 334 to pre-compute anticipated-query answers for the most predictable queries (so wake-time serve avoids even the recurrent step for those). The c' artifact from Plan 334 sits alongside the recurrent state from Plan 154 — one is the compressed history, the other is the anticipated future.

**The curiosity inversion bridges them:** Plan 334's `PredictabilityScorer` trait lets a consumer swap in a curiosity-inversion scorer where `p = sigmoid(α·(curiosity_ref − curiosity(c)))`. Curiosity here is the forecast residual of a KARC-style forecaster (Plan 308) — when the context is on-manifold (low curiosity, well-forecast), Plan 334 pre-computes aggressively; when it's off-manifold (high curiosity, novel), Plan 334 defers to wake-time fresh compute. Plan 154 has no such query-aware gate — it consolidates unconditionally.

📖 Plan 334: [`.plans/334_sleep_time_query_anticipator_primitive.md`](../.plans/334_sleep_time_query_anticipator_primitive.md). Architecture entry: [`.docs/02_architecture.md`](02_architecture.md) § Sleep-Time Query Anticipator. Examples: [`sleep_time_01_basic.rs`](../crates/katgpt-core/examples/sleep_time_01_basic.rs) + [`sleep_time_02_curiosity_inversion.rs`](../crates/katgpt-core/examples/sleep_time_02_curiosity_inversion.rs).
