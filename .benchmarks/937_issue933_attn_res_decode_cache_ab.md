# Bench 937 — Issue 933: `apply_attn_res` decode, push-time Σv² cache + prepared score weight

**Status:** DONE 2026-10-13 (katgpt-rs `12e031cff`). Exact (bit-identical) refactor, measured win; no flag needed (same function, same reductions, same summation order).

## What changed (katgpt-transformer `attn_res.rs`)

- `AttnResBlockState` caches each summary's `Σ v²` at `push` (it depends only on the summary, not the sublayer weights; it was re-reduced for every summary on every sublayer call). `residuals` is now private, read via `residuals()`, so a summary can only change via `push`/`clear`, which keep the cache in step. No external crate touched the field (grepped katgpt-rs + riir-*).
- The weighted-sum loop uses zipped slices (no bounds checks), same order.
- New `apply_attn_res_prepared`: skips the per-call `score_weight` re-fold for callers whose weights are frozen. `apply_attn_res` = re-fold + prepared, so every consumer keeps its current call and gets the cache for free.

## Gates

- G1 parity: `live_and_prepared_are_bit_identical_to_the_pre_933_kernel` — **bit-identity** (stronger than the issue's 1e-6 bar) vs a verbatim frozen copy of the pre-933 kernel at N ∈ {0,1,2,8,32}, 16 prefixes each. Stale-view canary `the_sum_sq_cache_follows_clear_and_repush` (clear + re-push into recycled pool slots; also asserts the re-push really moved the answer).
- G1 spec: `attn_res_g1_spec_match` 4/4, attn_res lib tests 7/7.
- G4: `kimi_k3_g4_alloc_free` 2/2 in **dev** (the release run reports a green 0 — the alloc gate is debug/alloc_tracking-gated); includes the full Kimi-K3 forward zero-alloc steady state. `kimi_k3_layer_diff` 1/1 release.
- Loud-zero: shared Issue-723 `ab_median_ratio` (per-iteration input churn, output consumed on both arms, named failure on a vanished arm).

## G2 — interleaved median-of-ratios, `--release`, d=1024, 31 rounds × 2000 iters

Box: M3 Max, AC power, load avg 2.2–2.5 (1-min), sibling agents idle-polling. Two consecutive runs:

| N summaries | ref ns | live/ref (run 1 / run 2) | prepared/ref (run 1 / run 2) |
|---|---|---|---|
| 2 (Kimi-K3-0.40B shape) | ~600 | 0.836 / 0.913 | 0.732 / 0.788 |
| 8 | ~1590 | 0.724 / 0.752 | 0.686 / 0.719 |
| 32 (long-context shape) | ~5950 | 0.776 / 0.768 | 0.734 / 0.736 |

Per-round spread ≤ ±4% around each median. Medians only; the ab harness does not produce a p99, so p99 is **not measured** (T5's p99 column stays open in that sense).

Run: `cargo test -p katgpt-transformer --features transformer_attn_res --test bench_933_attn_res_decode_ab --release -- --ignored --nocapture --test-threads=1`

## T1 (online-softmax merge): declined on analysis

In single-token decode the AttnRes "query" is the learned, static per-sublayer `score_weight`, and the block state is per token. Each (sublayer, summary) pair is scored and accumulated exactly once per token, so a running (max, sum, acc) state has nothing to reuse: total work stays `O(L·N·d)`. The MSAR two-phase win is a batched-query (prefill) memory-access win. The real redundancy in decode was the per-call `Σ v²` and `score_weight` re-fold, which this bench removes exactly.

## Honest scope

The only `apply_attn_res` consumer is `src/kimi_k3/` (test-arch only), and this kernel is a sliver of a layer next to attention + MoE. The win is substrate hygiene, not a decode-throughput claim.
