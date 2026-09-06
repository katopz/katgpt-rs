# Issue 730: Deterministic prefill KV-offload double-buffer for the 4090 256K lane (SparDA R539 extraction)

**Status:** OPEN — PoC task, verify-first (T0 can close this as N/A)

**Filed by:** katgpt-rs Research 539 (arXiv:2606.04511 SparDA, No-GD advocate finding), 2026-09-06.

## The mechanism

At **prefill**, causal attention at layer l attends *every* prior chunk — so the next
layer's KV demand is a **known deterministic schedule**, not a prediction. A
double-buffer that streams layer l+1's KV slice of chunk c−1 from CPU→GPU over PCIe
**while computing layer l of chunk c** gets the prefetch/compute overlap with zero
learning, zero selection, zero Forecast.

Why this matters: at 256K context the prefill KV **write** volume alone exceeds 24GB
VRAM (R436's ≈67GB wall) — offload is unavoidable at prefill *regardless of
compression* (compression governs what you READ; nothing is compressed yet while
prefilling). Order-of-magnitude from that wall: ~256KB/token total KV → ~8MB/layer
per 2048-token chunk → fetch:compute ≈ 1:1 at the tail chunks → un-overlapped offload
≈1.5×'s the prefill wall; a double-buffer hides it. SparDA (the paper) trains a
Forecast projection for this overlap at DECODE; its own prefill gain is selection-cost
only. The prefill half of the problem needs no learner at all.

**This is the one mechanism that attacks the 256K wall directly** — complementary to
FlashMemory (R436, shipped, Bench 671): compression shrinks what you read at decode;
offload+overlap removes the prefill wall; the MRCR dense-memory failure case is where
the lossless offload lane becomes the fallback compression cannot serve.

## Constraints

- **DRY:** the prefetch pipeline MUST compose INTO the FlashMemory cold/hot pool
  promotion path (`flashmemory_sparse.rs`) — a second offload system beside it is a
  violation.
- **Decode is out of scope:** a 2048-token chunk is ~8MB ≈ 0.3ms PCIe vs ~11ms/token
  compute — bandwidth is not the decode wall; decode-side prefetch is bounded by
  cold-miss rate and is not armed here.
- G1 of any landed gate: prefetch changes TIMING, never SELECTION — logits
  bit-identical (`to_bits`).

## Tasks

- [ ] **T0 — verify the wall (close-as-N/A gate).** Count the dspark checkpoint's
  full-attention layers vs DeltaNet layers from the GGUF header and recompute
  KV-bytes/token. Bonsai is a DeltaNet hybrid; if only ~3 layers are full attention,
  the 67GB wall shrinks proportionally and this issue closes as N/A (the wall number
  came from R436's pre-hybrid accounting and is UNVERIFIED).
- [ ] **T1 — price the serialized baseline.** Simulated-offload chunked prefill at
  64K/128K: wall-clock with synchronous PCIe writeback per layer vs the GPU-resident
  baseline. Establishes whether the 1.5× tail-wall estimate holds on our box.
- [ ] **T2 — double-buffer prototype.** Prefetch stream + pinned staging, composed
  into the FlashMemory pool promotion path. If a persistent copy kernel is used:
  re-sweep CTA count on 4090 (never inherit H100's 16/32; B104 no-harm rule applies
  to any bucket table shipped).
- [ ] **T3 — G1 bit-identity.** `to_bits` logit identity, prefetch on vs off, both
  chunk orders (the Bench-734-class bit-identity gate).
- [ ] **T4 — G2 wall-clock.** Overlapped vs serialized at 128K/256K, ≥1.15×-class
  promote bar; interleaved median-of-ratios; GPU-exclusive window.
- [ ] **T5 — G8 MRCR-style A/B.** Compression-only vs compression+offload-fallback
  on a scattered-needle dense-retrieval fixture (the R436 §1.6 failure regime).

## Related

- Research 539 (full per-track distillation + panel)
- R436 / Issue 584 (closed — Benches 021–026 + 671 are the FlashMemory record)
- riir-train Issue 524 (the training-track sibling)
- External: SparDA (arXiv:2606.04511) Table 10 (overlap crossover), TierKV (KV tiering
  with overlapped prefetch — prior art for the shape)
