# Bench 717: linking_fold detector Option B — 50 ms @ n=2×1000 RESTORED (Issue 757)

**Status:** DONE — Issue 757 P0+P1+P3 complete; T2.1 closed by decision (below). GOAT gate re-run: G1/G2/G2b/G4/G5 ALL PASS.
**Feature:** `linking_fold_detector` — **OPT-IN** (unchanged; the T3.1 verdict keeps it opt-in — see §Verdict). `linking_fold_fold` — DEFAULT-ON (unchanged; fold latencies re-measured, all budgets met).
**Source:** [arXiv:2606.31856](https://arxiv.org/abs/2606.31856) "Low-dimensional topology of deep neural networks" (Ren & Lim, ICML 2026) · Research 391 · Plan 410 · Issue 050 (resolved Option A) · Issue 757.
**Box:** 4090 (i7-13700K), Windows, bench profile (release), `CARGO_TARGET_DIR=/tmp/lf757`, 2026-09-12.

## Baseline (pre-Option-B, same box, 2026-09-12 re-bench)

| Gate | Pre-757 | Budget |
|---|---|---|
| G2 detector @ n=2×200, d=8 (linked, thickness 0.05) | 115.74 ms (407 ms pre-CycleBounds) | ≤ 500 ms (audit, Issue 050 Option A) |
| Original Plan-410 target @ n=2×1000 | "minutes" extrapolated (brute O(β²)) | ≤ 50 ms — UNREACHABLE (the Option A rationale) |

Profile at n=2×200 (debug, pre-757 code): Phase 4 Gauss pair loop **99.4%** of wall (4602 ms full sweep; kNN 22.6 ms, basis 4.0 ms, PCA 0.1 ms). β=446/449 → 200,254 pairs; BB-skip rejected 84.5%; **witness found only at evaluated-pair #20,161** (cycle lens 45/55 vs the ~5-10 noise-loop population). The bottleneck was never the may_link sweep — it was (a) evaluated-pair quadrature on near-but-unlinked pairs before the witness and (b) the witness sitting deep in basis order.

## The Option B levers (all verdict-preserving, `linking_detector.rs`)

1. **Single-pass k-NN (T1.1)** — the old graph builder walked all O(n²) pairs TWICE (ε quantile pass + adjacency pass), each with a full O(n log n) sort per node. Now: one pass, `select_nth_unstable_by` per node (avg O(n), no sort), squared-distance comparisons throughout (sqrt never taken — monotone ⇒ same sets).
2. **Longest-first witness ordering (T1.2, measured-motivated)** — each basis sorted DESCENDING by cycle length; the witness is a long core-winding cycle, so it is evaluated in the first pairs instead of after ~20K near-but-unlinked pairs.
3. **Certified chunk-level Gauss pruning (T1.2)** — per-cycle SoA segments + chunk bounds (runs of 8 segments) built ONCE per detect run (the old path rebuilt SoA per PAIR); per chunk-pair `link_bound` (the Issue-050 bound at chunk granularity) skips provably-negligible chunk pairs with a rigorous rounding rule: true ∈ [total − bound_sum, total + bound_sum]; `hi < 0.5` → certified 0; `round(lo) == round(hi)` → certified integer (sign restored from the partial); else full-recompute fallback. δ = 0.25/n_chunk_pairs caps skipped mass.
4. **Y-cycle uniform grid (T1.2 "spatial batching")** — prunes the O(β_x·β_y) may_link sweep; conservative per-X reach `r_x + max_r_y + sqrt(P_x·max_P_y/(2π))` provably covers every pair the pairwise bound keeps.

Plus two fixes found on the way:
- **`max_cycles_per_cloud` semantics fixed** — the pre-757 code kept the SHORTEST N cycles (and its comment claimed "short cycles dominate the integral") while the config doc promised MEDIAN-closest; measured reality: the witness is LONG, so shortest-N retained exactly the noise loops and dropped witnesses. Fixed to the documented median-closest (still opt-in, default 0, still NOT correctness-safe — any cap is heuristic).
- **Bench fixture scaling** — a fixed thickness 0.05 under-samples the ring once spacing 2π/n < 0.05 (n ≳ 300): cycles go jagged, quadrature error swamps the ±1 integer, and BOTH pre-757 and 757 read link=0 (verified on the PRE-757 code in a clean worktree: n=100 ✓ / 200 ✓ / **300 ✗**). The sweep + G2b fixtures use `min(0.05, π/n)`. The G1/G2 gate fixtures at n=200 keep 0.05 for historical comparability.

## Results (2026-09-12, release)

Scaling sweep (`LINKING_SCALING=1`, median of 3, d=8, thickness π/n for n>~300):

| n (per cloud) | linked ms | link | unlinked ms | link |
|---|---|---|---|---|
| 100 | 1.06 | +1 | 1.69 | 0 |
| 200 | 3.76 | −1 | 4.37 | 0 |
| 500 | 18.07 | +1 | 18.40 | 0 |
| 1000 | 27.89 | −1 | 59.91 | 0 |

GOAT gate re-run (`cargo bench -p katgpt-core --features linking_fold --bench bench_410_linking_fold_goat`):

| Gate | Measured | Budget | Verdict |
|---|---|---|---|
| G1 correctness smoke (n=200) | Hopf −1 / unlinked 0 / fold unlinks | exact | ✅ |
| G2 detector @ n=2×200 | **3.691 ms** (min 3.636) | ≤ 500 ms | ✅ (31.6× under the 115.74 ms pre-757 number) |
| **G2b detector @ n=2×1000 (NEW)** | **28.279 ms**, \|link\|=1, thickness π/n | ≤ 50 ms | ✅ — **the restored Plan-410 original budget** |
| G2 fold Abs/Gelu D=8 | 14.26 / 16.65 ns | ≤ 50 ns | ✅ |
| G2 fold Abs/Gelu D=64 | 17.40 / 26.40 ns | ≤ 500 ns | ✅ |
| G4 alloc (fold hot path, `linking_fold_alloc_check`) | 0 allocs | 0 | ✅ |
| G5 determinism | link=−1 ×3; fold bit-identical ×100 | exact | ✅ |
| G3 no-regression | default lib suite 2035 passed (unchanged); clippy --lib --features clean | — | ✅ |

New correctness tests: `cap_keeps_median_closest_drops_both_tails`, `pruned_integral_matches_full_on_random_pairs` (200 randomized near/far pairs — pruned ≡ full integers, caught a real sign-drop bug in the certified branch during development), `longest_first_grid_keeps_hopf_verdict`.

## T2.1 — closed by decision

`min_cycle_len` default stays **4**: the detector detects correctly across n=100..1000 on the resolvable fixture with the default, and raising it (paper §I.1 used 30 on CIFAR-10-scale data) would be tuning to synthetic fixtures — the right value is data-profile-dependent (real consumers set it per their sampling density; documented in the config).

## T3.1 — verdict: KEEP OPT-IN

The original 50 ms @ n=2×1000 budget is now MET, so perf no longer blocks promotion — but promotion still has no consumer: the only runtime consumer in the workspace (`riir-engine/latent_functor/linking_fold_bridge.rs` `LinkingFoldCorrector`) consumes the **fold** (default-on), not the detector. The detector remains zero-cost-unless-invoked opt-in; promotion re-opens when an audit-cadence consumer (sleep-cycle hook, shard-retrieval gate) exists.

## Known limits (honest)

- The unlinked-separated sweep at n=2×1000 is 59.9 ms — over the 50 ms original budget (which G2b pins to the LINKED path). The worst case remains an unlinked-tangled pair (no early exit; every grid candidate pays ≥ the may_link check).
- The under-sampled-fixture pathology (link=0 on a genuinely linked cloud when thickness > spacing) is inherent to graph-cycle Gauss detection at that regime — documented, not fixed here; the mitigation is fixture/data hygiene (scale thickness with sampling density).
