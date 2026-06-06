# Research 176: Vortex — Programmable Sparse Attention (Consolidated)

**Date:** 2026-06
**Source:** arXiv:2606.06453 — Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents
**Status:** Consolidated GOAT Verdict
**Supersedes:** Research 001 (Vortex Fusion Analysis), Research 175 (Vortex Distilled)
**Related Research:** 071 (DashAttention), 086 (RTPurbo), 100 (EGA), 145 (Wall Attention), 070 (GDN2), 042 (SP-KV), 048 (MuxDdTree), 028 (HLA)
**Related Plans:** 106 (DashAttention), 126 (RTPurbo), 139 (EGA), 173 (Wall Attention), 196 (VortexFlow Trait), riir-ai Plan 229 (Meta-Routing Game AI)

---

## Paper TL;DR

Vortex introduces a two-stage programming model for sparse attention that decomposes *all* sparse algorithms into: (1) query-independent cache preprocessing (`forward_cache`) and (2) query-dependent block selection (`forward_indexer`). On top of this, the paper makes three empirical discoveries:

1. **Channel concentration**: Out of 128 QK channels in 8 groups of 16, **only g3 and g7 matter for routing**. Masking other groups is harmless; masking g3 or g7 is catastrophic. The routing signal is structurally concentrated in ~25% of channels (32/128 dims).

2. **AI-agent discoverable policies**: LLM agents (Claude Sonnet 4.6, GPT-5) generated 20+ diverse sparse attention algorithms when given Vortex's composable operator catalog, with the best achieving 3.46x throughput over full attention. Value-energy gating (`⟨q, centroid⟩ · ‖v‖`) emerged as non-obvious but strong.

3. **vTensor paged abstraction**: Non-contiguous memory via page tables, enabling prefix caching. This is CUDA/GPU-specific.

**Results:** Up to 4.7x speedup with MLA sparse attention, 3.46x with agent-discovered policies, 1.3-1.62x from stochastic radix top-k. Validated on Qwen3-4B and Qwen3-8B.

**Executive verdict:** Of 7 fusion ideas, **2 are GOAT** (VortexFlow trait, Channel-Aware Routing), **1 is worth pursuing** (Meta-Routing Bandit), and **4 should be skipped or deferred** (Paged Game State, Stochastic Top-K, MLA-Style Compressed LoRA, WGSL Kernel Fusion).

---

## What Transfers from CUDA → CPU-SIMD / wgpu

| Vortex Concept | CUDA Original | Our Stack | Transfer |
|---|---|---|---|
| Two-stage decomposition | CUDA kernel scheduling | Rust trait abstraction | ✅ **Perfect** — math, not CUDA |
| Channel concentration g3/g7 | Masking experiment on GPU | SIMD dot product reduction | ✅ **Direct** — smaller dot product |
| Operator catalog (Table 1) | Triton/Codegen DSL | Rust trait methods | ✅ **Better** — monomorphized |
| Block size 16, top-k≈125 | GPU block scheduling | CPU cache-line aligned blocks | ✅ Works — 16×128 = 2KB fits L1 |
| Meta-routing (bandit selection) | Offline AI agent discovery | Online `BanditPruner` selection | ✅ **Better** — real-time adaptation |
| Value-energy gating | `⟨q, centroid⟩ · ‖v‖` on GPU | Same on CPU-SIMD | ✅ Direct |

## What Doesn't Transfer and Why

| Vortex Concept | Why Not |
|---|---|
| vTensor paged memory | GPU KV-cache-specific. Our CPU cache is contiguous `Vec<f32>` per layer. Five-tier system handles tier migration. |
| Stochastic radix top-k | Saves GPU microseconds over thousands of blocks. Our argmax is over 3-10 items on CPU — already O(n≤10). |
| MLA-Compressed LoRA insight | General principle ("don't compress routing signal") already followed. Specific MLA insight about attention routing doesn't transfer to LoRA adapter routing. `WallMLA` handles our MLA integration. |
| WGSL/CubeCL kernel fusion | Building a DAG-based fusion compiler is 3-6 month detour. Existing hand-fused kernels (`sgmv_lora_a_fused`, `sgmv_lora_b_fused`) cover high-value cases. |

---

## Creative Fusion Analysis — 7 Ideas

### Idea A: VortexFlow Trait — Unify Sparse Routing Behind One Trait

**GOAT: ✅ YES** | Gain 4/5 | Fit 5/5 | Risk 2/5

**Core insight:** All sparse attention algorithms decompose into `forward_cache` (query-independent) + `forward_indexer` (query-dependent). This is a **mathematical property of attention**, not a CUDA trick. BlockTopK computes centroids in stage 1, scores in stage 2. Quest computes min/max envelopes in stage 1, envelope products in stage 2. H2O accumulates attention scores in stage 1, uses running importance in stage 2. The decomposition mirrors information retrieval: build index offline, query online.

**Mapping:**

```rust
/// Two-stage sparse KV block routing (Vortex decomposition).
///
/// Stage 1 (forward_cache): Query-independent block summary computation.
/// Called once per cache update, reused across all decode steps.
///
/// Stage 2 (forward_indexer): Query-dependent block selection.
/// Called per decode step with current query.
trait VortexFlow: Send + Sync {
    /// Block summary to maintain alongside KV cache.
    type Cache;

    /// Query-independent: compute/maintain block summaries.
    /// Called when new KV blocks are appended or modified.
    fn forward_cache(
        &self,
        cache: &mut Self::Cache,
        keys: &[f32],        // [block_size, head_dim]
        block_idx: usize,
    );

    /// Query-dependent: score blocks and select top-k.
    /// Returns selected block indices + routing weights.
    fn forward_indexer(
        &self,
        query: &[f32],       // [head_dim]
        cache: &Self::Cache,
        n_blocks: usize,
        top_k: usize,
        scratch: &mut RoutingScratch,
    ) -> RoutingDecision;
}

struct RoutingDecision {
    /// Selected block indices.
    blocks: Vec<usize>,
    /// Routing weights (from entmax, softmax, or sigmoid).
    weights: Vec<f32>,
}
```

**Existing code → trait impls:**

| Algorithm | `forward_cache` | `forward_indexer` |
|---|---|---|
| BlockTopK | `Mean(c["k"])` → centroids | `GeMM(centroids, q^T)` → `topK` |
| DashEntmax | `ChunkSummaryQuery` projection | `entmax_1p5` on logits → support |
| Quest | `Max/Min(c["k"])` envelope | `Maximum(q*max, q*min)` → `topK` |
| ValueEnergyGate | Centroid + mean ‖V‖ | `⟨q, centroid⟩ · ‖v‖` |
| DoubleSparse | `Gather(c["k"])` channels | `GeMM(channels, q_gathered)` → `topK` |

**Why GOAT:**
- **Unification**: DashAttention (`dash_attn/routing.rs`), ShardKV, and future sparse attention all plug into one trait. Adding a new algorithm is just a new `impl VortexFlow`.
- **Zero-cost abstraction**: Rust monomorphization = no vtable overhead. Generic over `VortexFlow` in the hot path compiles to direct calls.
- **Composability**: `RoutingScratch` pattern already exists in `dash_attn/routing.rs` (lines 38-65). The trait formalizes it.
- **Enables meta-routing** (Idea E): With all algorithms behind one trait, bandit selection becomes trivially composable.
- **Enables channel pruning** (Idea B): `forward_cache` is the natural place to extract routing channels.

**Performance:** `forward_cache` runs once per block append (not in hot loop). `forward_indexer` runs per decode step — same as current `score_blocks_entmax`. Zero regression.

**katgpt-rs only.** This is an inference-time routing abstraction. riir-ai doesn't do KV cache routing.

**Implement this first.** It's the foundation for everything else.

---

### Idea B: Channel-Aware Routing — Extract Only Critical Channels

**GOAT: ✅ YES (conditional on validation)** | Gain 5/5 | Fit 4/5 | Risk 3/5

**Core insight:** Vortex Section 6.3 proves that out of 128 QK channels split into 8 groups of 16, **only g3 and g7 matter for routing**. Masking any other group: harmless. Masking g3 or g7: catastrophic. The routing signal is *highly concentrated* in ~25% of channels (32/128 dims). This holds across model sizes (Qwen3-4B and Qwen3-8B), suggesting it's a structural property of transformer attention.

**This is the most important non-systems insight in the paper.** You don't need the full query for routing — you need a small projection of it.

**Mapping:**

Currently `score_blocks_entmax` does full-dimension dot product:
```rust
let dot: f32 = query.iter().zip(s_ref.iter()).map(|(a, b)| a * b).sum();
```

If routing only needs channels g3 and g7 (32 out of 128 dims):

```rust
struct ChannelAwareCache {
    /// Routing channels only: [n_blocks, routing_dim]
    /// routing_dim = ~25% of head_dim (discovered per model)
    routing_keys: Vec<f32>,
    /// Full keys for actual attention: [n_blocks, head_dim]
    full_keys: Vec<f32>,
}
```

**SIMD gain:**
```rust
// Before: dot(query[0..128], centroid[0..128]) = 16 SIMD ops
// After:  dot(query[48..64], centroid[48..64]) + dot(query[112..128], centroid[112..128])
//         = 4 SIMD ops (NEON: 2 vmlaq_f32)
```

**Auto-discovery:** Don't hardcode g3/g7. Run the Vortex masking experiment at calibration time to discover which channels matter for *this specific model*, then store as a bitset.

**Implementation strategy:**
1. Add `RoutingChannelDiscovery` to calibration pipeline
2. At calibration: mask channel groups, measure accuracy delta, find critical groups
3. Store discovered routing channels in `DashAttnConfig`
4. `forward_indexer` uses routing channels for scoring, full channels for attention
5. Fallback: full-dimension routing if calibration fails

**Performance:** 4x faster routing dot product. `head_dim=128` → `routing_dim=32`. Memory overhead: 25% on centroids — negligible. Calibration: one-time ~5 min per model.

**Risk:** g3/g7 is Qwen3-specific. May not transfer to other architectures. Needs calibration infrastructure. Mitigated by fallback.

**katgpt-rs** (routing at inference). riir-ai doesn't benefit — GPU LoRA training doesn't do per-block routing.

**High-value, medium-risk.** Implement after Idea A. The validation experiment is cheap to run.

---

### Idea C: Paged Game State (vTensor for Game Memory)

**GOAT: ❌ NO** | Gain 2/5 | Fit 1/5 | Risk 4/5

**Verdict:** vTensor pages are homogeneous (same size, type, kernel). Game state pages are heterogeneous (board, history, reward, features — different types, sizes, access patterns). Five-tier memory system already handles tier migration. vTensor's page sharing (prefix caching) doesn't apply across game episodes. Fundamental abstraction mismatch.

---

### Idea D: Stochastic Top-K for Bandit Arms

**GOAT: ❌ NO** | Gain 1/5 | Fit 2/5 | Risk 2/5

**Verdict:** Our bandit/top-k is over 3-10 elements on CPU (nanoseconds). Exact argmax is already O(n≤10). Stochastic radix top-k saves GPU microseconds over thousands of blocks — different scale domain entirely. The math doesn't transfer.

---

### Idea E: Meta-Routing Bandit — Online Algorithm Selection

**GOAT: ⚠️ MAYBE** | Gain 4/5 | Fit 4/5 | Risk 3/5

**Core insight:** With all routing behind `VortexFlow`, a bandit selects the best routing algorithm per decode step. This is the **online analog** of Vortex's offline AI-agent discovery. Vortex's agents discovered value-energy gating as non-obvious but strong — our meta-router would discover such things in real-time.

**Mapping:**

```rust
/// Meta-router: bandit selects which VortexFlow implementation to use.
struct MetaRouter<F: VortexFlow> {
    /// Candidate routing policies.
    policies: Vec<Box<dyn VortexFlow<Cache = F::Cache>>>,
    /// Bandit selects policy index per decode step.
    bandit: BanditPruner,
}
```

Each decode step:
1. Bandit selects a policy (arm)
2. Policy scores blocks → `RoutingDecision`
3. Attention runs on selected blocks
4. Reward signal: `acceptance_rate * latency_improvement`
5. Bandit updates arm statistics

**This is genuinely novel.** Vortex used AI agents for *offline* algorithm discovery. We'd use bandits for *online* algorithm selection — the routing strategy adapts per-request, per-context-length, per-load-level.

**Landing:** Both codebases. katgpt-rs: KV block routing per decode step. riir-ai: LoRA adapter routing strategy per game frame.

**Risks:**
- **Cold start**: Bandit needs exploration episodes before convergence. Bad routing during exploration hurts P99 latency.
- **Reward signal**: `acceptance_rate` from speculative decoding or per-token perplexity — neither trivially available.
- **Policy count**: With 3-5 policies, UCB1 converges in ~50 episodes. With 20+, much longer.

**Practical first step:** Implement 3 policies (BlockTopK, Entmax, ValueEnergyGate), wire into `BanditPruner`, measure convergence on benchmark. Fallback: pin best policy from offline benchmarks.

**Positive if policies differ:** If different policies are optimal for different input types (code vs. math vs. conversation), meta-routing captures this automatically. Overhead: ~50ns per decode step — negligible.

**Pursue after Ideas A and B are in place.** Start with 3 policies, validate convergence.

---

### Idea F: MLA-Compressed LoRA Routing

**GOAT: ❌ NO** | Gain 2/5 | Fit 2/5 | Risk 3/5

**Verdict:** The MLA insight is about *attention routing* in compressed-KV architectures, not LoRA adapter routing. The general principle ("don't compress away routing signal") is already embedded in our tier placement logic. `WallMLA` already handles MLA integration. The analogy to game-phase information is strained — RoPE is a specific positional encoding, not analogous to game-state features.

---

### Idea G: WGSL Kernel Fusion

**GOAT: ❌ NO (defer)** | Gain 3/5 | Fit 3/5 | Risk 5/5

**Verdict:** Building a DAG-based fusion compiler for CubeCL is 3-6 month detour. Existing hand-fused kernels (`sgmv_lora_a_fused`, `sgmv_lora_b_fused`) cover the two highest-value operations. Vortex's fusion system works because they have ~20 operators that compose in many ways. We have ~5 LoRA operators that compose in one fixed pattern. Revisit if operator count grows significantly.

---

## GOAT Verdict Table

| Idea | Description | Landing | GOAT | Implement? |
|---|---|---|---|---|
| **A** | VortexFlow trait | katgpt-rs | ✅ YES | **Phase 1** |
| **B** | Channel-aware routing | katgpt-rs | ✅ YES* | **Phase 2** (*conditional on validation*) |
| **E** | Meta-routing bandit | Both | ⚠️ MAYBE | **Phase 3** (needs convergence proof) |
| C | Paged game state | riir-ai | ❌ NO | Skip |
| D | Stochastic top-k | katgpt-rs | ❌ NO | Skip |
| F | MLA-compressed LoRA | riir-ai | ❌ NO | Skip |
| G | WGSL kernel fusion | riir-ai | ❌ DEFER | Revisit if ops grow |

---

## Integration with Existing Systems

### DashAttention (Plan 106, Research 071)

`score_blocks_entmax` in `dash_attn/routing.rs` currently does: chunk logits → `entmax_1p5` → support extraction. This is `forward_indexer` only, with `ChunkSummaryCache` handling cache-level work. VortexFlow formalizes the separation. `EntmaxRouter` becomes `impl VortexFlow for EntmaxRouter` wrapping existing code — zero behavioral change.

### Wall Attention (Plan 173, Research 145)

Wall's diagonal forget gates produce per-channel retention scores. Task 6 (Plan 173) already proposes "use gate-derived forgetfulness scores for block-level routing decisions." VortexFlow's `forward_cache` is the natural place to consume Wall prefix sums: when all channels of a key have decayed below threshold → skip block. The Wall + DashAttention integration (Plan 173 T6) becomes `impl VortexFlow for WallAwareRouter`.

### EGA Energy-Gated Attention (Plan 139, Research 100)

EGA gates attention by spectral energy of key tokens. This is a routing signal. ValueEnergyGate from Vortex's agent discoveries (`⟨q, centroid⟩ · ‖v‖`) is closely related. Both can be `impl VortexFlow` — EGA uses energy, ValueEnergyGate uses value norms. The meta-router (Idea E) can learn which one is better per context.

### RTPurbo Retrieval Heads (Plan 126, Research 086)

RTPurbo identifies "always-on" vs "dynamic" retrieval head dimensions. This is analogous to Vortex's channel concentration discovery. If g3/g7 corresponds to retrieval-critical dimensions, the two findings reinforce each other. `RoutingChannelDiscovery` could leverage RTPurbo's head classification as a warm start.

### BanditPruner (Plan 030)

`BanditPruner<P>` wraps any `ScreeningPruner` and adds Q-value learning. The `MetaRouter` is `BanditPruner` applied at the algorithm-selection level instead of the token-selection level. Same Thompson/UCB1 machinery, different granularity. The existing `BanditStrategy::EpsilonGreedy { epsilon, decay }` works directly for meta-routing.

### MuxDdTree (MUX, Research 048)

`MuxDdTree` in `crates/katgpt-core/src/mux/dd_tree.rs` already has superposition-based tree expansion with `MuxSpanPruner`. VortexFlow's `RoutingDecision` (blocks + weights) is the routing analog of `MuxDdTree`'s span selection (top-k peaks). The two are orthogonal: MuxDdTree selects tokens, VortexFlow selects KV blocks. But the pattern is identical — VortexFlow is MuxDdTree applied at the cache level.

### HLA (Plan 057, Research 028)

HLA's symmetric second-order accumulators (`SK`, `CQV`, `mQ`) and asymmetric accumulators (`PKV`, `mK`) maintain compressed state across positions. VortexFlow's `forward_cache` does the same at the block level — compressed block summaries that enable routing without full KV access. HLA's `simd_outer_product_accumulate` (Plan 060) pattern applies to block centroid computation in `forward_cache`.

### Plasma Ternary (Plan 148)

`TernaryWeights` in `types.rs` (bit-plane packed {-1, 0, +1}) with `simd_ternary_matvec` on NEON/AVX2. If routing projections are quantized to ternary, channel-aware routing could use the same SIMD paths. The `blocks64` field maps naturally to block-level cache storage — each routing centroid row is `blocks64 × 64` bits packed.

---

## Key Numbers from Paper

| Metric | Value | Implication |
|---|---|---|
| Block size | 16 tokens | 16 × head_dim = 2KB per block, fits L1 |
| Top-k default | ~125 blocks | ~125 × 2KB = 250KB working set, fits L2 |
| Routing channels | 2/8 groups (g3, g7) | 32/128 dims = 25% of dot product |
| Agent-discovered policies | 20 per model | We start with 3, bandit selects |
| Best agent speedup | 3.46x vs full attention | Routing alone (no compression) |
| MLA sparse speedup | 4.7x | Requires MLA architecture |
| Calibration time | ~5 min per model | One-time masking experiment |

### Repo-Verified Algorithm Benchmarks (Qwen3-1.7B, B200)

From `flow/ALGORITHMS_RESULTS.md` — all 8 built-in algorithms at identical knobs (block_size=16, topk=29, ratio=0.0625, skip_layers=[0]):

| Algorithm | RULER Accuracy | Throughput (tok/s) | Key Observation |
|---|---|---|---|
| `block_sparse_attention` | 0.98 | 2793 | Centroid + GeMM top-k baseline |
| `gqa_block_sparse_attention` | **1.00** | 2750 | GQA-aware centroid routing |
| `gqa_quest_sparse_attention` | **1.00** | 2837 | Quest with GQA — fastest accurate |
| `lserve_sparse_attention` | **1.00** | 2783 | LServe-style per-head routing |
| `masked_quest_sparse_attention` | **1.00** | 2782 | Quest with masked out-of-range |
| `centered_block_sparse_attention` | 0.98 | 2859 | Centroid minus global mean — fastest |
| `running_avg_block_sparse` | 0.99 | 2383 | EMA accumulator (Save/Load) — slower due to radix cache disabled |
| **`venergy_gated_centroid`** | **1.00** | 2781 | **⟨q, centroid⟩ · ‖v‖ — our Idea B variant** |

**Key insight:** `venergy_gated_centroid` matches best-in-class accuracy (1.00) and throughput (~2.8k tok/s). Value-energy gating is **not** a niche algorithm — it's competitive with the best. This strengthens the case for implementing it as a VortexFlow variant in Phase 1.

### The Actual "CUDA Trick" — Triton Batched-Paged-Ragged MatMul

From `indexer/triton_kernels/matmul_impl.py` — `mm_bpr_kernel`:
- Handles `[B, G, D] × [S, C, D]` matmul where S is packed (ragged, different length per B)
- Uses persistent caching of query tiles (`current_x_idx` tracking to avoid re-loading)
- `tl.constexpr` for G, C, D enables Triton JIT specialization
- The "trick" is fusing ragged iteration + paged gather + batched GEMM into one kernel

**Why this doesn't transfer:** Our CPU decode path does `dot(query[head_dim], cache_key[head_dim])` per block — contiguous memory, no page tables, no ragged batching. The Triton kernel solves a GPU memory layout problem (non-contiguous paged KV cache). On CPU, the equivalent is just a loop of SIMD dot products over a contiguous `Vec<f32>` — already optimal.

### The Compiler (graph.py + compile.py)

The JIT fusion system is a full DAG compiler:
- `indexer/compiler/graph.py`: Builds compute DAG from vFlow operations
- `indexer/compiler/compile.py`: Generates Triton kernels from DAG
- `indexer/compiler/triton_impl/`: Per-operation Triton codegen
- `indexer/compiler/cuda_impl/`: CUDA fallback for specific ops
- `indexer/compiler/custom_impl/`: Hand-tuned specializations (e.g., Schedule.S)

**Confirmed:** Building this in CubeCL/WGSL would be a 3-6 month compiler project. Our hand-fused WGSL kernels cover the high-value cases. Skip (Idea G).

---

## Implementation Priority

```
Phase 1: VortexFlow trait + BlockTopK + ValueEnergyGate impls  (Foundation)
Phase 2: Channel-Aware routing SIMD calibration                 (4x routing speedup if validated)
Phase 3: Meta-Routing Bandit                                    (Online algorithm selection)
```

Feature gate: `vortex_flow` — default-OFF until GOAT proof passes.

### Skip

| Idea | Reason |
|---|---|
| C (Paged Game State) | Abstraction mismatch — vTensor solves KV-specific paging, not heterogeneous game state |
| D (Stochastic Top-K) | Wrong scale — our top-k is over 3-10 elements on CPU, not thousands of GPU blocks |
| F (MLA-Compressed LoRA) | General principle already followed, specific MLA insight doesn't transfer to LoRA |
| G (WGSL Kernel Fusion) | Requires building a CubeCL fusion compiler — 3-6 month detour for marginal gains |

---

## Key Takeaways Beyond the 7 Ideas

1. **Operator catalog (Table 1)** is an excellent reference for what operations a sparse attention system needs. Our `VortexFlow` trait should expose the same set: `Mean`, `Max`, `Min`, `GeMM`, `Multiply`, `Add`, `Softmax`, `topK`, `Load/Save` (stateful).

2. **The 18-hour autonomous optimization loop** (Section 6.1.2) converged to block top-k with tuned hyperparameters, not fundamentally new algorithms. This suggests the algorithm space is well-explored and implementation quality matters more than novelty. Supports our "few good policies + bandit selection" approach.

3. **MLA sparse attention** (Section 6.2, 4.7x speedup) is already partially addressed by our `WallMLA` integration. Vortex's rope-aware scoring validates our approach of keeping positional and content information decoupled.

4. **Block size 16 with top-k≈125** is a robust default across model scales. This is a concrete tuning recommendation we can use immediately for DashAttention's `ChunkSummaryCache`.

---

## TL;DR

Vortex's two-stage decomposition (`forward_cache` + `forward_indexer`) is a **mathematical property of sparse attention**, not a CUDA trick. It transfers directly to our Rust/CPU-SIMD stack as the `VortexFlow` trait. The channel concentration discovery (g3/g7 = 25% of channels carry routing signal) could give 4x routing speedup. The meta-routing bandit is a novel online adaptation of Vortex's offline AI-agent discovery.

**Repo verification confirms:** `venergy_gated_centroid` (our Idea B variant) achieves 1.00 accuracy and competitive throughput — it's not a niche algorithm. The actual "CUDA trick" (Triton batched-paged-ragged matmul) solves GPU memory layout problems we don't have on CPU.

**Implement A → B → E. Skip C, D, F, G.** Feature gate `vortex_flow`, default-OFF until GOAT proof.

What doesn't transfer: vTensor paging (wrong domain), stochastic top-k (wrong scale), MLA insight (already followed), kernel fusion (too expensive to build). These aren't bad ideas — they're solving problems we don't have or already handle differently.
