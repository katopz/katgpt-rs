# Bench 662 — Batched width-N Metal forward (Issue 660 task 5)

**Date:** 2026-08-17
**Issue:** `660` — task 5 (batched width-N forward)
**Device:** M3 Max (local), GPU-exclusive (no compute consumer; Unity Editor + Zed active per the exemption)
**Config:** `Config::micro()` (vocab=27, n_embd=16, n_head=4, n_layer=1, mlp_hidden=64, block_size=16) — overhead-dominated by design; measures dispatch/sync cost, not compute.

## Change

`GpuBackend` gains `forward_batch(tokens, pos)` — N tokens processed in ONE
command buffer with a single commit + wait, returning logits for EVERY
position (the MTP verification shape; flat row-major
`out[i * vocab..(i + 1) * vocab]`). This was the task-5 follow-up the
single-submit refactor (Bench 661) unblocked.

Mechanics:

- The per-token dispatch body (embedding → layers → lm_head) was extracted
  ONCE into `GpuFrame::encode_token`, shared verbatim by `forward` and
  `forward_batch` — identical kernels, values, and per-token dispatch order;
  only the submit grouping differs.
- **Per-slot buffers, not shared**: pos/seq_len differ per token, and CPU
  writes to a shared `contents()` buffer all land before GPU execution —
  every dispatch would see the last value. Each of the N tokens gets its own
  pre-allocated embedding-row, pos-scalar, seq_len-scalar, and logits slot
  (`MAX_FORWARD_BATCH = 16`), written exactly once before the single commit.
- Causality within the batch: token i's `kv_store` lands at `pos + i` and its
  attention dispatches read exactly `[0, pos + i]` — encoder serialization
  makes earlier batch tokens' KV visible, the identical guarantee the
  single-token path relies on (Bench 661).
- Cross-token scratch reuse (x/q/k/v/…) is safe by the same serialization —
  consecutive tokens reuse buffers exactly like sequential separate forwards.
- CPU fallback (uncompiled backend) loops `katgpt_forward` per token.
  `InferenceBackend` trait untouched — promote to a trait method when an MTP
  consumer actually needs backend-agnostic batching.

## G1 — bit-identity vs sequential (PASS)

`test_gpu_forward_batch_matches_sequential`: the sequence `[0, 1, 3, 7, 5]`
run as 5 sequential single-token GPU forwards vs ONE `forward_batch` call —
**every logit bit-identical at every position** (`to_bits()` equality), plus
a follow-up token at pos 5 run on each cache afterwards (bit-identical —
probes KV-cache state divergence the per-position rows alone cannot).
Also vs CPU: cosine ≥ 0.999 at every position
(`test_gpu_forward_batch_matches_cpu`).

## G2 — width-N latency (interleaved protocol, PASS)

Same-binary interleaved pairs (2 discarded warmups + 5 measured, alternating
seq→batch / batch→seq within pairs, per-pair µs/token ratio; global warm-up
loop first; 50 iterations per arm). Release build:

| width | seq (µs/token) | batch (µs/token) | median speedup | ratio spread |
|---|---|---|---|---|
| 2 | 352.5 | 272.7 | **1.35×** | 1.29–1.35 |
| 5 | 348.7 | 211.9 | **2.19×** | 1.65–2.28 |
| 8 | 347.3 | 139.4 | **2.37×** | 2.08–2.49 |

Width-8 batch runs at **139.4 µs/token — 2.5× below the post-661
single-token path** (255–282 µs/token, Bench 661).

The width-2 saturation is structural, not noise: with per-token encode+exec
cost E and one commit+wait W, the ratio is `N·(E+W)/(N·E+W)` — at E≈130 µs,
W≈210 µs the width-2 ceiling is ≈1.35× and width-8 ≈2.4×, matching the
measurement. Raising the ceiling further requires amortizing E itself
(GEMM-shaping the matmuls with M=N rows — changes the MSL pipeline set,
out of scope here; the kernels are untouched per Bench 661's scope note).

## G3 — no-regression (PASS)

- `cargo test -p katgpt-backend --features gpu_inference --lib`:
  **28/28 debug, 28/28 release** (23 pre-existing + 5 new: batch-vs-sequential
  bit-identity, batch-vs-CPU cosine, CPU fallback bit-identity, empty batch,
  interleaved bench with per-width release-only floors 1.1 / 1.5 / 1.5).
- `cargo clippy -p katgpt-backend` (`gpu_inference` + default, `--all-targets`):
  0 warnings.
- `cargo check --workspace --features gpu_inference`: clean.

## Scope notes

- The `InferenceBackend` trait is untouched — `forward_batch` is an inherent
  `GpuBackend` method (no consumer needs backend-agnostic batching yet).
- `MAX_FORWARD_BATCH = 16` (per-slot buffers pre-allocated at `compile()`).
- MTP/speculative work on Metal is unblocked: verification-shaped forwards
  (N candidate tokens → N logit rows) now cost one submit instead of N.

## Verdict

**GOAT PASS.** G1 bit-identical (incl. KV-cache follow-up probe), G2 1.35× /
2.19× / 2.37× at widths 2/5/8 (139.4 µs/token at width 8), G3 clean. Issue
660 task 5 closes; the issue is fully resolved.
