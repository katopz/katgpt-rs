# Bench 661 — Single-submit Metal forward (Issue 660)

**Date:** 2026-08-17
**Issue:** `660` — single-submit (graph-fused) Metal forward
**Device:** M3 Max (local), GPU-exclusive (no other compute consumer)
**Config:** `Config::micro()` (vocab=27, n_embd=16, n_head=4, n_layer=1, mlp_hidden=64) — overhead-dominated by design; this config measures dispatch/sync cost, not compute.

## Change

`GpuBackend::forward` previously encoded the forward into **9 command buffers with 9 `wait_until_completed()` CPU syncs per pass** (1 embedding + 1 per-layer prologue + `n_head` per-head buffers + 4 per-layer epilogue + 1 lm_head) — roughly `5 + 4·n_layer + n_head·n_layer` round trips per token.

The refactor encodes the whole forward into **one command buffer with a single commit + wait**:

- Encoder order within a command buffer IS the GPU-side ordering (dispatches within a compute encoder serialize with memory visibility — the same guarantee the old waits provided; the code already relied on it for matmul→relu→matmul in one encoder).
- The one real mid-forward CPU↔GPU dependency — the `x→xr2` pre-norm save, a CPU `ptr::copy_nonoverlapping` via `contents()` — became a **blit-encoder copy inside the command buffer** (byte-for-byte identical; this CPU read was what forced the per-block syncs).
- All `n_head` attention heads now share one compute encoder: head h's value-sum completes before head h+1 overwrites the shared `scores_buf` (identical ordering semantics to the old per-head waits).

## G1 — bit-identity vs the old path (PASS)

A temporary test printed exact f32 bit patterns of all logits over the fixed sequence `[0, 1, 3, 7, 5]` (shared KV cache, progressive positions) on both builds:

- before (HEAD `3967cad9` worktree): 5 positions × 27 logits, bit-exact hex captured
- after (refactored tree): **identical in every bit** (`diff` empty after dedup)

No kernel, dispatch order, or buffer binding changed — the only semantic delta is WHERE the `x→xr2` copy executes (CPU ptr copy → GPU blit), which is a byte copy either way.

## G2 — width-1 decode latency (interleaved protocol, PASS)

Protocol per the repo's measurement discipline (Bench 666/Issue 658 corrections): 2 warmup pairs discarded, 5 measure pairs, alternating A→B / B→A order within pairs, per-pair ratio as primary metric. Both binaries built `--release --features gpu_inference`; A = worktree at HEAD, B = refactored tree.

| Pair | order | before (µs/token) | after (µs/token) | ratio after/before |
|---|---|---|---|---|
| 1 | A→B | 2063.1 | 262.5 | 0.1272 |
| 2 | B→A | 2021.6 | 282.4 | 0.1397 |
| 3 | A→B | 2025.1 | 257.1 | 0.1270 |
| 4 | B→A | 2018.8 | 275.9 | 0.1367 |
| 5 | A→B | 2053.3 | 255.3 | 0.1243 |

**Median ratio 0.1272 → the single-submit forward is 7.86× faster** on the overhead-dominated micro config. Before/after spreads are tight (±1% / ±5%), consistent with a real effect.

Reference lines: `CPU: 1.2 µs/token, GPU: 1897.1 µs/token` (before) → `CPU: 1.2 µs/token, GPU: 241.4 µs/token` (after). At n_embd=16 the GPU still loses to CPU by ~200× — expected: this config's purpose is to expose the per-token dispatch/sync tax (what blocked MTP per Bench 656), not to beat CPU on toy compute. The ~240 µs residue is one command-buffer submit+wait (~5 encoders for n_layer=1) + the bench harness's per-iteration `ForwardContext::new`/`MultiLayerKVCache::new`.

## G3 — no-regression (PASS)

- `cargo test -p katgpt-backend --features gpu_inference --lib`: **23/23 pass** (incl. `test_gpu_forward_matches_cpu`, `test_gpu_forward_multi_token_matches_cpu`, `test_gpu_forward_deterministic`, `test_goat_gpu_forward_matches_cpu`).
- `cargo clippy -p katgpt-backend --features gpu_inference --all-targets`: 0 warnings.

## Scope notes

- `bench_mtp_metal_batch_floor` (Bench 656) is unaffected — it drives kernels directly with its own single-submit interleaving, not through `GpuBackend::forward`. This change makes the production forward path consistent with that kernel-level analysis.
- The follow-on batched width-N forward (Issue 660 task 5) stays deferred per the issue's "only then" ordering.
- Further encoder-count reduction (a copy kernel to avoid the blit's encoder split) is possible but changes the MSL pipeline set — out of scope.

## Verdict

**GOAT PASS.** G1 bit-identical, G2 7.86× (median interleaved ratio), G3 clean. The dispatch-overhead artifact behind the Metal MTP loss is removed from the production forward path on this backend.
