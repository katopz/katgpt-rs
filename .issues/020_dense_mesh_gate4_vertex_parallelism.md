# Issue 020: DenseMesh Gate 4 — Vertex Parallelism for width-4 ≤ 2.5× bound

**Status:** CLOSED (improved but not met — Path B pooled dispatch landed, ratio dropped from 2.76–3.04× to 2.53–2.95×; median ~2.60×, still above the 2.5× paper bound. Remaining gap is CPU-vs-GPU kernel fusion, documented as DenseNode trait extension follow-up.)

**Closure rationale (2026-06-20):** Path B (pooled scratch/output dispatch in `forward_layer_parallel_pooled` + `transformer::forward_batched` infrastructure) landed. New measured ratio at `Config::small_target()`: **2.53–2.95×** (5-run median **~2.60×**), down from Path A's 2.76–3.04×. The pooled path eliminates the ~8 per-call allocations (4 `MeshScratch` + 4 `DenseHidden`) that Path A was doing on every hidden-layer transition — zero allocator traffic in steady state after warmup. `forward_batched` is implemented as correct, tested infrastructure (prefill semantics: N tokens at consecutive positions, shared KV cache, bit-identical to sequential `forward`), but full matmul-level batching is NOT wired because the `DenseNode` trait exposes only single-token `forward_dense` and `TransformerNode`'s `weights`/`ctx_pool`/`cache_pool` are private — fusion requires a trait extension that was out of scope for the file-constrained Path B task. The remaining ~0.1× gap to 2.5× is the architectural difference between CPU rayon dispatch (4 parallel tasks, ~5μs spawn overhead each) and the paper's assumed GPU kernel fusion (1 fused batched matmul). Closing as improved-but-not-met; true 2.5× requires either a Metal/GPU backend or a `BatchedDenseNode` trait extension.

**Source**: Plan 266 Phase 7 gate 4 measurement — `tests/dense_mesh_goat_gates.rs::test_dense_mesh_gate4_hard_bound_width4_measured`
**Priority**: Medium (blocks true GOAT promotion of `dense_mesh`; gate is currently `#[ignore]` and documents the gap)
**Blocked**: No
**Depends**: Nothing (rayon already in tree; transformer.rs is local)

## Problem

The paper's ≤ 2.5× latency bound at width 4 assumes **vertex parameter sharing + parallel execution** — the 4 hidden nodes in a layer share one LLM and execute in parallel (batched GPU forward or rayon on CPU).

katgpt-rs's current `LayerwiseTopology::forward` runs all hidden nodes **sequentially**. As a result, the measured ratio at `[1, 4, 1]` topology is:

```
baseline (1×fwd)     │    0.20μs   │  1.00x
mesh[1,4,1] (5×fwd)  │    1.87μs   │  9.27x   ← measured, paper bound 2.5x
```

This is the expected sequential cost (5 forwards × ~1 vanilla + aggregation overhead). The bound is **unreachable** without parallel execution.

## Reproduction

```bash
# Gate 4 measurement (ignored by default — measurement, not pass/fail)
cargo test --release --features dense_mesh --test dense_mesh_goat_gates \
  test_dense_mesh_gate4_hard_bound_width4_measured -- --nocapture --include-ignored
```

See `.benchmarks/266_densemesh_goat.md` for full numbers.

## Proposed fix (two paths, both likely needed)

### Path A — Rayon across hidden nodes (smaller change)

Modify `LayerwiseTopology::forward` to use `rayon::scope` when the hidden layer width ≥ `gpu_width_threshold` (default 4). Each hidden node borrows `&TransformerWeights` shared, with its own `ForwardContext` + `MultiLayerKVCache` per thread.

Expected speedup at width 4: ~2.5× (4 parallel threads → ~1.5× wall-clock after overhead). Ratio drops from 9.27× → ~3.7×. Still over 2.5×.

**Cost:** ~50 LoC in `src/dense_mesh/topology.rs`. Thread-safety analysis on `DenseNode` (currently `&self` — good, no mutation needed).

### Path B — Batched forward in transformer.rs (larger change)

Add `forward_batched(ctx, weights, cache, tokens: &[usize], pos, config) -> Vec<&mut [f32]>` that processes N tokens at once, amortising KV cache writes and matmul setup.

Expected speedup at width 4: ~1.2× on top of rayon (better memory locality). Combined with Path A, ratio drops to ~3× → 2.5×.

**Cost:** ~200 LoC in `src/transformer.rs` (new entry point + re-organisation of the per-token loop). Risk of regressing existing forward paths.

### Recommendation

Start with **Path A** (rayon) — small, isolated, measurable. If ratio still > 2.5× after Path A, file a follow-up for Path B.

## Acceptance criteria

- [x] Gate 4 test un-ignored (removed `#[ignore]`; now runs at both `draft` and `small_target` scale)
- [ ] Measured ratio at `[1, 4, 1]` topology ≤ 2.5× vanilla forward — **Path B landed (pooled dispatch + `forward_batched` infrastructure). New measurement: 2.53–2.95× (5-run median ~2.60×) at `small_target` (n_embd=64). Down from Path A's 2.76–3.04×. Still above 2.5× — remaining gap is CPU rayon spawn overhead vs paper's assumed GPU kernel fusion. Full `forward_batched` matmul fusion blocked on `DenseNode` trait extension (out of scope for file-constrained Path B).**
- [x] No regression in `prof_dense_mesh` aggregation/forward scaling tests — 5/5 pass; width=16/width=1 = 7.04× (threshold <16×) after Path B pooling
- [x] No data race in `MultiLayerKVCache` — per-thread `Mutex` pools indexed by `rayon::current_thread_index()`; verified by `test_transformer_node_parallel_forward_is_safe` (8 parallel workers, bit-identical outputs)

## Status update (Path A applied 2026-06-19)

**Path A (rayon vertex parallelism) is complete** behind opt-in `MeshConfig::enable_vertex_parallelism` (default `false`). Dispatch triggers when `width_next >= gpu_width_threshold`. `TransformerNode` now holds per-thread `Mutex` pools for `ForwardContext` + `MultiLayerKVCache` (pool size = `available_parallelism()`), so each rayon worker locks only its own slot — uncontended, no data race.

**Result:** at `Config::small_target()` (n_embd=64, ~60μs/fwd), width-4 ratio dropped from ~5× (sequential) to **2.76–3.04×** — Path A beat sequential by ~1.8×. Still above the 2.5× paper bound.

At `Config::draft()` (sub-μs forwards), rayon spawn overhead dominates and Path A regresses — expected at micro-bench scale; the win only materializes once the per-forward work exceeds thread-pool overhead (~5μs), which matches the issue's own caveat.

## Status update (Path B applied 2026-06-20)

**Path B (pooled scratch/output dispatch + `transformer::forward_batched` infrastructure) is complete.** Two pieces landed:

1. **`transformer::forward_batched`** (`src/transformer.rs`) — a new public entry point that processes N tokens at consecutive positions (`pos_start..pos_start+N-1`) in a single call. Writes per-token logits into a flat `ForwardContext::batch_logits` buffer (resized once, reused across calls — no per-token alloc). Returns `Vec<&mut [f32]>` with one vocab-sized slice per token. Prefill semantics: token `i` sees K/V of tokens `0..i` in the same batch. Verified bit-identical to sequential `forward` on a shared cache by `test_forward_batched_matches_sequential` (5 tokens × vocab=4096, max_diff < 1e-5).

2. **`forward_layer_parallel_pooled`** (`src/dense_mesh/topology.rs`) — when `enable_vertex_parallelism && width_next >= VERTEX_BATCH_THRESHOLD (4)`, the hidden-layer transition draws `Vec<MeshScratch>` and `Vec<DenseHidden>` from `LayerwiseTopology::scratch_pool` / `output_pool` instead of allocating them per call. This eliminates the ~8 allocations Path A was doing every iteration. The pool is a `Mutex<Vec<...>>` locked once per forward (not per rayon task) — negligible contention.

**Why full `forward_batched` matmul fusion is NOT wired:** the topology holds `node: Box<dyn DenseNode>`, and the `DenseNode` trait exposes only `forward_dense(&self, input, layer_idx, scratch) -> DenseHidden` (single-token). `TransformerNode`'s `weights` / `ctx_pool` / `cache_pool` are private. To call `forward_batched` from the topology would require either (a) extending `DenseNode` with a `forward_dense_batched` method, or (b) adding public accessors to `TransformerNode`. Both are outside the file-constrained Path B scope (`src/transformer.rs`, `src/dense_mesh/topology.rs`, `tests/dense_mesh_goat_gates.rs`, `.issues/020_*.md`). Filed as DenseNode trait extension follow-up.

**Result:** at `Config::small_target()` (n_embd=64), 5-run measurement of width-4 ratio: **2.53×, 2.56×, 2.60×, 2.81×, 2.95×** (median ~2.60×). Down from Path A's 2.76–3.04×. One run hit 2.53× — within measurement noise of the 2.5× bound — but the median is still above. The remaining ~0.1× gap is the architectural difference between CPU rayon dispatch (4 parallel tasks, ~5μs spawn overhead each, no kernel fusion) and the paper's assumed GPU batched forward (1 fused matmul).

**Verdict:** Path B improved the ratio honestly but did not meet 2.5×. Closing as improved-but-not-met. True 2.5× on CPU likely requires either (a) a `BatchedDenseNode` trait extension so the topology can invoke `forward_batched` directly on the shared vertex, or (b) a Metal/GPU backend that fuses the 4 hidden forwards into one kernel launch.

## Path B follow-up (deferred)

The remaining ~0.5× gap to the 2.5× bound needs **Path B — batched transformer forward** in `src/transformer.rs` (~200 LoC): a `forward_batched(ctx, weights, cache, tokens: &[usize], pos, config) -> Vec<&mut [f32]>` entry point that processes N tokens at once, amortizing KV cache writes + matmul setup. Expected additional ~1.2× on top of Path A → ~2.5× combined.

This is a separate, larger change touching `transformer.rs` internals. File as a new issue when the `dense_mesh` feature sees real workload that justifies the risk.

## References

- Research: `.research/234_DenseMesh_Latent_Node_Network.md` (gate 4)
- Plan: `.plans/266_densemesh_latent_node_network.md` Phase 7
- Benchmark: `.benchmarks/266_densemesh_goat.md`
- Paper: arXiv:2505.12741 §3.3 (vertex parameter sharing) + §3.1.3 (cost model)
