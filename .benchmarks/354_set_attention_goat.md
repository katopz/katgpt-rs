# Benchmark 354: Cross-Datapoint Set Attention — G3+G4 Perf Gate

**Date:** 2026-07-01
**Plan:** [katgpt-rs/.plans/354_cross_datapoint_set_attention_primitive.md](../.plans/354_cross_datapoint_set_attention_primitive.md)
**Research:** [katgpt-rs/.research/354_Cross_Datapoint_Set_Attention_NPT.md](../.research/354_Cross_Datapoint_Set_Attention_NPT.md)
**Source paper:** [arXiv:2106.02584](https://arxiv.org/pdf/2106.02584) — Kossen et al., NeurIPS 2021 (NPT)
**Bench:** [`crates/katgpt-core/benches/set_attention_bench.rs`](../crates/katgpt-core/benches/set_attention_bench.rs)
**Features:** `set_attention` (opt-in)

## GOAT gate results

| Gate | Criterion | Target | Result | Verdict |
|---|---|---|---|---|
| **G1** | Permutation equivariance (bit-exact up to float reorder) | max\|Δ\| < 1e-6 over 10 random perms | PASS (see `tests/set_attention_g1_g5.rs`) | ✅ |
| **G2** | Identity-floor meaningfulness (2-cluster) | cluster means preserved, separation preserved, bounded | PASS | ✅ |
| **G3** | Latency at N=64, d=8, k=4 | < 25 µs (prod target, 2000× tick headroom) | **21.96 µs** | ✅ |
| **G4** | Zero allocations (dense path) | 0 allocs/call | **0** | ✅ |
| **G5** | Sigmoid-not-softmax (lonely query) | sharper β reduces lonely motion | PASS | ✅ |

**Supplementary (top-k path):** 1 allocation per call (documented — the `Vec` for index sort). Sparse path is not the hot path for N ≤ 64; deferred to a future zero-alloc top-k API if a real crowd-scale use case demands it.

## Phase 3 T3.2 — Sparse top-k scale sweep (N=1000 target)

**Target:** N=1000, d=8, k=4, top_k=16 → < 100 µs/call.

**Result: FAIL (35× over target).** Exact top-k cannot meet the target because the top-k path still computes ALL N² per-pair sigmoid scores to know which peers are in the top-k — it only reduces the *accumulation* step (O(k_max·d) vs O(N·d)), not the *scoring* step (O(N²·k) in both paths).

| N | dense (µs/call) | top_k=16 (µs/call) | top-k speedup |
|---|---|---|---|
| 128 | 87.86 | 64.82 | 1.36× |
| 256 | 336.42 | 243.76 | 1.38× |
| 512 | — | 940.65 | — |
| 1000 | — (est. ~5000) | **3516.80** | — |

**Honest root cause:** the `topk_accumulate` implementation scores all N peers per query (`O(N²·k)`) then does a full sort (`O(N log N)` per query). Even with `select_nth_unstable` (partial sort), the O(N²·k) scoring dominates. For N=1000, k=4: 4M FMAs ≈ 1ms even with 4-wide SIMD — the 100µs target is impossible for ANY exact method at N=1000.

**Verdict:** T3.2 FAILS. The only path to sub-100µs at N=1000 is T3.3 (LSH-based approximate top-k — locality-sensitive hashing avoids scoring all N candidates). T3.3 stays deferred until a real N>100 use case emerges. **Current production (Plan 355 G9) runs 100 NPCs on the DENSE path at 75.7µs/tick (6.6× headroom under 500µs gate) — no current use case demands sparse top-k.** The top-k code ships as-is for callers who want sparser accumulation at moderate N (the 1.36–1.38× speedup at N=128–256 is real), but it is NOT a crowd-scale solution.

## Latency scale sweep

The dense path is O(N²·k + N·d²). Empirical measurements (release build, macOS):

| N | µs/call | Notes |
|---|---|---|
| 16 | 1.75 | Solo patrol / small zone — meets the speculative 5µs target |
| 32 | 5.93 | Typical NPC zone — at the 5µs threshold |
| 64 | 21.96 | Gate target (prod-pass, speculative-fail) |
| 128 | 85.70 | Large zone — top-k path recommended |
| 256 | 333.54 | Very large crowd — top-k required |

## Honest assessment

- **G3 production-pass:** 21.96 µs at N=64 is 0.04% of the 50 ms tick budget (2200× headroom). Production-safe.
- **G3 NPC-zone-deferred:** the speculative 5 µs target is met at N ≤ 16 but missed at N=32+ on the dense path. Closing this gap needs SIMD (the inner k=4 dot product and d=8 accumulation are perfect for NEON/AVX2). Deferred until riir-ai Plan 355 G9 (production-latency at 100-NPC zones) demands it.
- **G4 zero-alloc PASS:** the dense path performs 0 heap allocations in steady state, verified via the counting allocator (codebase convention from `bench_313_ac_prefix_goat.rs`).

## Reproduction

```bash
cd /Users/katopz/git/katgpt-rs
cargo bench -p katgpt-core --features set_attention --bench set_attention_bench -- --warm-up-time 0 --measurement-time 2 --sample-size 10

# Correctness gates (G1, G2, G5):
cargo test -p katgpt-core --features set_attention --test set_attention_g1_g5
```

## Verdict

**Plan 354 Phase 2 GOAT gate: PASS (G1, G2, G3-prod, G4, G5).** The speculative 5µs-at-N=64 target (G3-NPC) is deferred to a SIMD optimization follow-up; the production target (25µs at N=64, 2000× tick headroom) is met.

**PROMOTED to default-on 2026-07-01.** Promotion condition (Plan 355 G6 fusion adds value) passed: G6 fusion cosine sim <0.95 (fusion produces qualitatively different attention than identity). Additional evidence: G7 crowd stability <5% drift over 100×2000 ticks, G9 production latency 75.7µs/tick at 100 NPCs (6.6× headroom under 500µs gate). G8 collective inference FAILED (Super-GOAT→GOAT) — averaging cannot amplify detection; that's a use-case limitation, NOT a primitive defect. The validated selling point is crowd coherence (belief sync, noise reduction, contextual awareness).

## TL;DR

Cross-Datapoint Set Attention open primitive (`set_sigmoid_attention_into`) passes all 5 GOAT gates: G1 permutation equivariance, G2 identity-floor meaningfulness, G3 latency (21.96µs at N=64, production-pass), G4 zero-alloc, G5 sigmoid-not-softmax correctness. The speculative 5µs-at-N=64 SIMD target is deferred. Latency scales O(N²): N=16→1.75µs, N=32→5.93µs, N=64→22µs. **DEFAULT-ON since 2026-07-01** after Plan 355 G6/G7/G9 passed (G8 FAILED — averaging cannot amplify detection; use-case limitation, not primitive defect). Validated selling point: crowd coherence.

**Phase 3 T3.2 (sparse top-k at N=1000) FAILED honestly:** 3517µs vs 100µs target. Exact top-k cannot beat O(N²) scoring — the topk path only reduces accumulation, not scoring. T3.3 (LSH) is the only viable crowd-scale path, deferred until a real N>100 use case emerges. Current production (G9, 100 NPCs) uses the dense path at 75.7µs/tick — no demand for sparse top-k yet.
