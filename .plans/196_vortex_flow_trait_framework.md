# Plan 196: VortexFlow Trait Framework — Composable Sparse Routing + Channel-Aware SIMD

**Branch:** `develop`
**Depends on:** Plan 106 (DashAttention ✅), Plan 173 (Wall Attention ✅), Plan 139 (EGA ✅), Plan 030 (BanditPruner ✅)
**Research:** `.research/176_Vortex_Programmable_Sparse_Attention_Consolidated.md` (supersedes 001, 175)
**Feature flag:** `vortex_flow` — default-ON (GOAT proof passed ✅ 72/72 tests, 3/3 criteria)
**Status:** ✅ Complete — GOAT PASSED, promoted to default-ON

---

## Problem

1. **No unified routing abstraction** — DashAttention's `score_blocks_entmax`, RTPurbo's low-dim scoring, and potential future sparse algorithms (BlockTopK, Quest, ValueEnergyGate) each implement their own block selection. Adding a new algorithm requires touching the hot decode path directly.

2. **Full-dimension routing is wasteful** — `score_blocks_entmax` computes `dot(query[0..head_dim], centroid[0..head_dim])` but Vortex proves only ~25% of channels carry routing signal. For `head_dim=128`, we compute 4x more than needed.

3. **No online algorithm selection** — Different input types (code, math, conversation) may benefit from different routing algorithms, but there's no mechanism to switch. The routing strategy is hardcoded at config time.

## Architecture

```
VortexFlow trait (src/dash_attn/vortex_flow.rs)
├── type Cache
├── forward_cache(cache, keys, block_idx)     — query-independent
└── forward_indexer(query, cache, top_k)      — query-dependent

Implementations:
├── EntmaxRouter         — wraps existing score_blocks_entmax
├── BlockTopKRouter      — centroid + GeMM top-k
├── QuestRouter          — min/max envelope + product scoring
├── ValueEnergyRouter    — centroid · ‖v‖ gating
├── ChannelAwareRouter   — routes on discovered critical channels (Phase 2)
└── MetaRouter           — bandit selects policy (Phase 3)

Data flow:
┌─ Prefill/Cache Update ────────────────────────────────────────┐
│                                                                │
│  KV block appended → forward_cache(block_keys)                │
│    ├─ EntmaxRouter: ChunkSummaryQuery projection               │
│    ├─ BlockTopKRouter: Mean(keys) → centroid                   │
│    ├─ QuestRouter: {Max(keys), Min(keys)} → envelope          │
│    ├─ ValueEnergyRouter: centroid + mean(‖V‖)                  │
│    └─ ChannelAwareRouter: keys[groups..] → routing_keys       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │
         ▼ cache updated
┌─ Decode Step ──────────────────────────────────────────────────┐
│                                                                │
│  query arrives → forward_indexer(query, cache, top_k)         │
│    ├─ EntmaxRouter: entmax_1p5(logits) → support              │
│    ├─ BlockTopKRouter: dot(centroids, q) → top_k              │
│    ├─ QuestRouter: max(q*max, q*min) → top_k                  │
│    ├─ ValueEnergyRouter: ⟨q, centroid⟩ · ‖v‖ → top_k         │
│    ├─ ChannelAwareRouter: dot(routing_keys, q[g3,g7]) → top_k│
│    └─ MetaRouter: bandit picks policy → delegate               │
│                                                                │
│  → RoutingDecision { blocks, weights }                         │
│  → Attention runs on selected blocks only                      │
└────────────────────────────────────────────────────────────────┘
```

### Alignment with Existing Systems

| System | Connection |
|---|---|
| **DashAttention (Plan 106)** | `EntmaxRouter` wraps existing `score_blocks_entmax`. Zero behavioral change. |
| **Wall Attention (Plan 173 T6)** | `WallAwareRouter` (future): consume Wall prefix sums for decay-aware block skipping. |
| **EGA (Plan 139)** | `ValueEnergyRouter` is the Vortex analog of EGA's energy gating. Meta-router can learn which is better per context. |
| **RTPurbo (Plan 126)** | `ChannelAwareRouter` discovery is analogous to RTPurbo's retrieval head identification. Cross-pollinate. |
| **BanditPruner (Plan 030)** | `MetaRouter` uses `BanditPruner<NoScreeningPruner>` for policy selection. Same UCB1/Thompson/EpsilonGreedy. |
| **MuxDdTree (MUX)** | Orthogonal — MuxDdTree selects tokens, VortexFlow selects KV blocks. Same pattern, different level. |
| **HLA (Plan 057)** | `simd_outer_product_accumulate` pattern applies to block centroid computation in `forward_cache`. |
| **Plasma Ternary (Plan 148)** | If routing projections quantize to ternary, `simd_ternary_matvec` applies. Future optimization. |
| **ScreeningPruner / ConstraintPruner** | Different granularity. Pruners operate at token level, VortexFlow at block level. Composable. |

---

## Phase 1: VortexFlow Trait + BlockTopK Impl (Foundation)

### Tasks

- [x] **T1: Define `VortexFlow` trait in `src/dash_attn/vortex_flow.rs`**
  - Associated type `Cache` for routing algorithm state
  - `fn forward_cache(&self, cache: &mut Self::Cache, keys: &[f32], block_idx: usize)`
  - `fn forward_indexer(&self, query: &[f32], cache: &Self::Cache, n_blocks: usize, top_k: usize, scratch: &mut RoutingScratch) -> RoutingDecision`
  - `fn cache_new(&self, n_blocks_capacity: usize) -> Self::Cache` — pre-allocate cache
  - Trait is `Send + Sync`, no generic bounds beyond that
  - Behind `#[cfg(feature = "vortex_flow")]`

- [x] **T2: Define `RoutingDecision` struct**
  - `blocks: Vec<usize>` — selected block indices
  - `weights: Vec<f32>` — routing weights (from entmax, softmax, or sigmoid)
  - Both pre-allocated with capacity = top_k
  - `fn clear(&mut self)` for reuse across decode steps

- [x] **T3: Define `RoutingScratch` reusable buffer**
  - `scores: Vec<f32>` — block scores (capacity = max_blocks)
  - `indices: Vec<usize>` — top-k index buffer
  - Already partially exists in `dash_attn/routing.rs` — extract and formalize

- [x] **T4: Implement `BlockTopKRouter`**
  - `forward_cache`: compute `Mean(keys[block_size])` → centroid per block. Store in `BlockTopKCache { centroids: Vec<f32> }` shape `[n_blocks, head_dim]`
  - `forward_indexer`: compute `dot(centroids[i], query)` for all blocks → `argtopk(scores, top_k)` → `RoutingDecision`
  - SIMD-optimize the centroid dot product using existing `simd_dot` or inline NEON/AVX2
  - This is the simplest possible VortexFlow impl — proves the trait works

- [x] **T5: Implement `EntmaxRouter` — wrap existing DashAttention**
  - `forward_cache`: delegate to existing `ChunkSummaryCache::update()`
  - `forward_indexer`: delegate to existing `score_blocks_entmax()`, wrap result in `RoutingDecision`
  - No behavioral change — the router is a thin wrapper over existing code
  - Validates that VortexFlow doesn't regress DashAttention

- [x] **T6: Implement `ValueEnergyRouter` — centroid · ‖v‖ gating**
  - `forward_cache`: compute centroid + mean ‖V‖ per block. Store in `ValueEnergyCache { centroids: Vec<f32>, v_energy: Vec<f32> }`
  - `forward_indexer`: `dot(centroids[i], query) * v_energy[i]` → top-k
  - Repo-verified: `venergy_gated_centroid` achieves RULER 1.00 accuracy and competitive throughput on Qwen3-1.7B
  - This validates the VortexFlow trait supports multi-signal routing (not just single-dot-product)

- [x] **T7: Wire `VortexFlow` into decode path** (Phase 2 wiring)
  - Add `vortex_router: Option<Box<dyn VortexFlow<Cache = DynRoutingCache>>>` to `Config` or `TransformerWeights`
  - In decode step: if `vortex_router.is_some()`, use `forward_indexer` for block selection instead of hardcoded DashAttention
  - If `None`: existing behavior (DashAttention hardcoded path)
  - Feature gate `vortex_flow` guards the new field

- [x] **T8: Unit tests for VortexFlow trait**
  - Test `BlockTopKRouter`: known keys → known centroids → known top-k
  - Test `EntmaxRouter`: existing DashAttention tests still pass via wrapper
  - Test `ValueEnergyRouter`: verify v_energy=0 gates out block, v_energy>0 passes centroid dot product
  - Test `RoutingDecision` clear/reuse (no re-allocation)
  - Test `RoutingScratch` buffer reuse across calls

- [x] **T9: Example `examples/vortex_01_block_topk.rs`**
  - Simulate KV cache with synthetic blocks
  - `BlockTopKRouter` selects top-k blocks for a synthetic query
  - Compare selected blocks vs full attention — verify routing quality
  - Feature gate: `vortex_flow`
  - Register in `Cargo.toml` with `required-features = ["vortex_flow"]`

---

## Phase 2: Channel-Aware Routing SIMD

### Tasks

- [x] **T10: Implement `RoutingChannelDiscovery` calibration**
  - Takes model + calibration data
  - For each of 8 channel groups (g0..g7): mask group, run routing, measure accuracy delta
  - If masking group g_k causes >5% accuracy drop → g_k is critical
  - Output: `RoutingChannelMask` bitset `[head_dim]` where bit = 1 if channel is routing-critical
  - Store discovered mask in config
  - One-time calibration per model

- [x] **T11: Implement `ChannelAwareCache`**
  - `routing_keys: Vec<f32>` — `[n_blocks, routing_dim]` where `routing_dim = sum(routing_channels)`
  - `full_keys: Vec<f32>` — `[n_blocks, head_dim]` for actual attention
  - Populated by `ChannelAwareRouter::forward_cache`: extract routing channels from each block's keys
  - Memory overhead: routing_dim/head_dim ≈ 25% additional storage on centroids

- [x] **T12: Implement `ChannelAwareRouter`**
  - `forward_cache`: extract routing channels from block keys → `ChannelAwareCache`
  - `forward_indexer`: `dot(routing_keys[i], query[routing_channels])` → top-k
  - SIMD optimization: routing_dim=32 means 4 NEON vectors (8 f32 each) vs 16 for full dim
  - Fallback: if `routing_channels` is `None`, delegate to full-dim routing via `full_keys`

- [x] **T13: SIMD-optimized channel-aware dot product**
  - NEON: `vld1q_f32` × 4 + `vmlaq_f32` × 4 + horizontal add (32 dims → 4 accumulators)
  - AVX2: `_mm256_loadu_ps` × 4 + `_mm256_fmadd_ps` × 4 + horizontal add
  - Scalar fallback for unknown architectures
  - Benchmark: routing dot product vs full dot product → expect 3-4x speedup

- [x] **T14: Calibration example `examples/vortex_02_channel_discovery.rs`**
  - Run `RoutingChannelDiscovery` on synthetic model
  - Print which channel groups are critical
  - Compare routing quality with vs without channel selection
  - Feature gate: `vortex_flow`

- [x] **T15: Phase 2 GOAT proof**
  - Benchmark: channel-aware routing latency vs full-dim routing latency
  - Target: ≥3x routing speedup, ≤1% quality regression
  - If target met: `vortex_flow` feature gate stays, channel-aware becomes the default router
  - If target not met: document finding, keep `BlockTopKRouter` as default

---

## Phase 3: Meta-Routing Bandit

### Tasks

- [x] **T16: Implement `MetaRouter`**
  - Owns `policies: Vec<DynPolicy>` — enum-dispatched policies
  - Owns `BanditPruner<NoScreening>` with `policies.len()` arms
  - `forward_cache`: delegates to ALL policies (maintains all caches)
  - `forward_indexer`: bandit selects policy arm → delegates to selected policy
  - Reward signal: `acceptance_rate * latency_improvement` per decode step
  - Strategy: `BanditStrategy::EpsilonGreedy { epsilon: 0.1, decay: 0.995 }`

- [x] **T17: Implement `DynRoutingCache` enum**
  - One variant per router cache type:
    - `BlockTopK(BlockTopKCache)`
    - `Entmax(EntmaxCache)`
    - `ValueEnergy(ValueEnergyCache)`
    - `ChannelAware(ChannelAwareCache)`
    - `Meta(Vec<DynRoutingCache>)`
  - `DynPolicy` enum dispatches `forward_cache`/`forward_indexer`/`cache_new`
  - `MetaRouter` needs all policy caches — variant is `Meta(Vec<DynRoutingCache>)`

- [x] **T18: Wire reward signal from speculative verification**
  - After `forward_indexer` → attention → speculative verification:
    - If token accepted: reward = `1.0 + latency_bonus`
    - If token rejected: reward = `0.0`
  - Feed reward to `MetaRouter.update_reward(arm, reward)`
  - Latency bonus: `(baseline_latency - actual_latency) / baseline_latency` ∈ [0, 1]
  - Implemented as `compute_reward(accepted, baseline_latency_ns, actual_latency_ns)`

- [x] **T19: Meta-routing benchmark example `examples/vortex_03_meta_router.rs`**
  - 3 policies: BlockTopK, Entmax, ValueEnergyRouter
  - Run 200 decode steps with synthetic queries
  - Bandit starts exploring (ε=0.3) → converges (ε→0.01)
  - Print: policy selected over time, average reward per policy, convergence point
  - Feature gate: `vortex_flow`

- [x] **T20: Phase 3 GOAT proof**
  - Benchmark: meta-router average latency vs best single policy latency
  - Target: meta-router latency ≤ best policy latency + 5% overhead
  - Target: meta-router discovers best policy within 50 decode steps

- [x] **T21: Update `.benchmarks/` with VortexFlow results**
  - Phase 1: BlockTopK routing latency vs DashAttention
  - Phase 2: Channel-aware routing latency vs full-dim routing
  - Phase 3: Meta-router convergence + latency

---

## Success Criteria

| Metric | Target | Measurement |
|---|---|---|
| `EntmaxRouter` wraps DashAttention | Zero behavioral regression | Existing DashAttention tests pass |
| `BlockTopKRouter` routing quality | ≥90% overlap with full attention | Block overlap on calibration data |
| Channel-aware routing speedup | ≥3x vs full-dim routing | Micro-benchmark |
| Channel-aware quality regression | ≤1% accuracy delta | Calibration masking experiment |
| Meta-router convergence | Best arm identified in ≤50 steps | Q-value convergence |
| Meta-router overhead | ≤5% vs best single policy | Latency benchmark |
| No regression | All existing tests pass | `cargo test` without `vortex_flow` |
| Feature gate isolation | Zero impact when disabled | `cargo test` without feature |

## GOAT Gate → Default-ON Criteria

All three must pass:

1. **VortexFlow trait adds zero overhead** — `cargo bench` shows no regression in decode throughput when feature is disabled
2. **Channel-aware routing proves ≥2x speedup** — validated on at least one target model
3. **Meta-router converges reliably** — best arm found in ≤50 steps on 3/3 benchmark inputs

If all pass → change `vortex_flow` to default feature in `Cargo.toml`. If any fail → keep opt-in, document why.

**Result: All 3 passed.** `vortex_flow` promoted to default-ON.

## Files to Create/Modify

```
src/dash_attn/
├── vortex_flow.rs           # T1: VortexFlow trait + RoutingDecision + RoutingScratch
├── block_topk.rs            # T4: BlockTopKRouter + BlockTopKCache
├── entmax_router.rs         # T5: EntmaxRouter wrapper
├── value_energy.rs          # T6: ValueEnergyRouter + ValueEnergyCache (repo-verified)
├── channel_aware.rs         # T10-T13: RoutingChannelDiscovery + ChannelAwareRouter
├── meta_router.rs           # T16-T17: MetaRouter + DynRoutingCache
└── mod.rs                   # Add pub mod declarations behind feature gate

examples/
├── vortex_01_block_topk.rs  # T9
├── vortex_02_channel_discovery.rs # T14
└── vortex_03_meta_router.rs # T19

Cargo.toml                   # Feature gate: vortex_flow = []
```

---

## TL;DR

Three-phase implementation of Vortex's composable sparse attention in katgpt-rs:

1. **VortexFlow trait** — unify DashAttention, BlockTopK, ValueEnergy behind `forward_cache` + `forward_indexer`. Zero regression.
2. **Channel-aware SIMD** — if g3/g7 generalizes, 3-4x routing speedup from smaller dot products. Conditional on validation.
3. **Meta-routing bandit** — online algorithm selection via `BanditPruner`. Convergence proof needed.

Feature gate `vortex_flow`, default-OFF. Goes default-ON only if GOAT gate passes (zero overhead, ≥2x channel speedup, ≤50-step convergence).
