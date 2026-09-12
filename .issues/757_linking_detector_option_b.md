# Issue 757: linking_fold detector Option B — the 50 ms @ n=2×1000 remainder (Issue 050 follow-up)

**Status:** DONE (2026-09-12) — P0+P1+P3 complete, T2.1 closed by decision, T3.1 verdict KEEP OPT-IN. The original Plan-410 budget (≤50 ms @ n=2×1000) is RESTORED: 28.28 ms linked (G2b gate). Full evidence: [Bench 717](../.benchmarks/717_linking_detector_option_b.md).
**Research:** [katgpt-rs/.research/391](../.research/391_Low_Dimensional_Topology_Linking_Number.md) §6 · **Source:** [arXiv:2606.31856](https://arxiv.org/abs/2606.31856) "Low-dimensional topology of deep neural networks" (Ren & Lim, ICML 2026) · **Prior:** Plan 410 (executed 2026-07-07, no plan file — see Research 391 §6), Issue 050 (RESOLVED 2026-07-07 Option A: accept the recalibrated 500 ms @ n=2×200 audit-cadence budget).

`linking_fold_detector` (Algorithm 1: PCA-3D + ε-kNN + fundamental cycle basis + Gauss linking integral) is opt-in at audit cadence. Issue 050's Option A resolution accepted the recalibrated budget; the Option B remainder — algorithmic work toward the original 50 ms @ n=2×1000 plan budget — was referenced in the module doc and bench doc as "tracked" but had **no issue file** until this one.

Post-resolution perf work already landed: `CycleBounds::may_link` bounding-sphere early-skip (provable `gap² > P_C·P_D/(2π)` ⇒ integral rounds to 0 — correctness-safe) + full-SoA auto-vectorized quadrature. Re-benched 2026-09-12 (4090 box): **407 ms → 115.74 ms median @ n=2×200, d=8**; fold hot-path 16.25/15.76 ns @ D=8, 18.18/26.14 ns @ D=64; G1/G5 PASS.

**Measured profile driving Option B** (pre-757, n=2×200, debug): 99.4% of wall = the Gauss pair loop; BB-skip already rejected 84.5% of pairs; the WITNESS was found only at evaluated-pair #20,161 (long core-winding cycle, len 45/55, vs the ~5-10 noise-loop population). So the levers were witness ordering + per-pair quadrature cost, not the may_link sweep.

## Tasks

### P0 — Baseline + the missing bench doc — **DONE (Bench 717)**

- [x] **T0.1** Scaling curve measured (linked + unlinked-separated, n ∈ {100,200,500,1000}, d=8): 1.06/3.76/18.07/27.89 ms linked; 1.69/4.37/18.40/59.91 ms unlinked. Instrument: `LINKING_SCALING=1` env section in bench_410 + env-parametrized prof harness (`LINKING_PROF_N`). **Fixture finding:** a fixed thickness 0.05 under-samples the ring at n ≳ 300 (thickness > spacing 2π/n) — jagged cycles, quadrature error swamps ±1, link=0 — and this hits the PRE-757 code too (verified in a clean worktree: n=300 already missed). Sweep fixtures scale thickness `min(0.05, π/n)`.
- [x] **T0.2** Bench doc landed as [Bench 717](../.benchmarks/717_linking_detector_option_b.md) (numbering: .benchmarks highwater 716→717), including the pre-757 baseline numbers and the profile.

### P1 — Correctness-safe algorithmic work — **DONE (GOAT re-run ALL PASS)**

- [x] **T1.1 Single-pass k-NN** — one pass, `select_nth_unstable_by` per node (no full sorts), squared-distance comparisons (sqrt never taken; monotone ⇒ identical sets).
- [x] **T1.2 The three measured levers** (superset of the filed "spatial batching"; the may_link sweep was measured NOT to be the bottleneck — 0.5% — so the grid is one of three levers, all verdict-preserving):
  - **Longest-first witness ordering** — bases sorted DESC by length; witness evaluated in the first pairs (this is where most of the 31.6× came from).
  - **Certified chunk-level Gauss pruning** — per-cycle SoA + chunk bounds built ONCE per run (was: per PAIR); per-chunk-pair `link_bound` skipping with a rigorous [total−bound_sum, total+bound_sum] rounding rule + full-recompute fallback for the ambiguous band; pruned ≡ full pinned by 200 randomized-pair unit tests (which caught a real sign-drop bug in the certified branch during development).
  - **Y-cycle uniform grid** — conservative per-X reach `r_x + max_r_y + sqrt(P_x·max_P_y/(2π))` provably covers every pair the pairwise bound keeps.
  - **Bonus fix:** `max_cycles_per_cloud` semantics corrected to the DOCUMENTED median-closest (the code kept the SHORTEST N while claiming "short cycles dominate" — measured false; the witness is long). Unit-tested.
- [x] **T1.3 Re-gate** — G1 ✅ / G2 @2×200 **3.69 ms** (was 115.74) ✅ / **NEW G2b @2×1000 linked 28.28 ms ≤ 50 ms with |link|=1** ✅ (the restored original budget, now an enforced bench row) / G4 alloc ✅ / G5 determinism ✅ / G3: default lib suite 2035 passed unchanged + clippy `--lib --features linking_fold` clean.

### P2 — Optional levers — **T2.1 closed by decision; T2.2 unchanged**

- [x] **T2.1** `min_cycle_len` default stays **4** (decision): detection is correct across n=100..1000 on the resolvable fixture at the default; raising it (paper used 30 on CIFAR-scale data) would be tuning to synthetic fixtures — the right value is data-profile-dependent (config doc records the guidance).
- [x] **T2.2** `max_cycles_per_cloud` stays opt-in non-default — now with the documented median-closest semantics actually implemented (see T1.2 bonus fix).

### P3 — Verdict — **DONE: KEEP OPT-IN**

- [x] **T3.1** Perf no longer blocks promotion (50 ms met), but there is still **no detector consumer**: the only runtime consumer workspace-wide (`riir-engine/latent_functor/linking_fold_bridge.rs`, `LinkingFoldCorrector`) consumes the FOLD (default-on), not the detector. Stays opt-in, zero-cost-unless-invoked; promotion re-opens when an audit-cadence consumer exists (sleep-cycle hook, shard-retrieval gate, healer-surface PoC below).

## Follow-up tracked here (fusion idea, novelty TBD — not a task)

- [-] **Healer-surface consumer exploration**: linking detection on (rule ↔ code-shape) embedding clusters in the riir-rag span-embedding space (PCA-3D) as a retrieval top-1 confusion predictor — the paper's CIFAR-10 analogue (linking consistency ↔ confusion, Spearman r ≈ 0.48). Novelty TBD: the healer corpus is discrete `AstChunker` spans, not smooth manifolds — the k-NN-cycle machinery's manifold assumption is unverified there. With G2b met, the perf precondition for this PoC no longer blocks; any promotion of this idea requires a PoC on the oracle-labeled retrieval fixtures (riir-clippy `retrieval_eval`) first; no plan until that PoC exists.

## Provenance

- Filing commit fixed the stale-surface drift (Cargo comments ×2, module-doc table). Execution commit (this close-out): linking_detector.rs (single-pass kNN, ordering, chunk pruning, grid, cap fix, module docs), bench_410 (G2b row, scaling sweep, refreshed constants/docs), tests ×3 new, Bench 717, this file, Cargo feature comment, Research 391 §6 pointer.
- Side-finding filed separately: the slice_tca test module does not compile under `cargo test --release` without `alloc_tracking` (`crate::alloc` is `debug_assertions`-gated) — the Issue-741 profile×feature class; blocks release-mode test runs of the crate (found while running this issue's release profile harness).
