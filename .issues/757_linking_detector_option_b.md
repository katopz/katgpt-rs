# Issue 757: linking_fold detector Option B — the 50 ms @ n=2×1000 remainder (Issue 050 follow-up)

**Status:** Open — filed from the Research 391 closure session (2026-09-12); no tasks executed yet.
**Research:** [katgpt-rs/.research/391](../.research/391_Low_Dimensional_Topology_Linking_Number.md) §6 · **Source:** [arXiv:2606.31856](https://arxiv.org/abs/2606.31856) "Low-dimensional topology of deep neural networks" (Ren & Lim, ICML 2026) · **Prior:** Plan 410 (executed 2026-07-07, no plan file — see Research 391 §6), Issue 050 (RESOLVED 2026-07-07 Option A: accept the recalibrated 500 ms @ n=2×200 audit-cadence budget).

`linking_fold_detector` (Algorithm 1: PCA-3D + ε-kNN + fundamental cycle basis + Gauss linking integral) is opt-in at audit cadence. Issue 050's Option A resolution accepted the recalibrated budget; the Option B remainder — algorithmic work toward the original 50 ms @ n=2×1000 plan budget — was referenced in the module doc and bench doc as "tracked" but had **no issue file** until this one.

Post-resolution perf work already landed: `CycleBounds::may_link` bounding-sphere early-skip (provable `gap² > P_C·P_D/(2π)` ⇒ integral rounds to 0 — correctness-safe) + full-SoA auto-vectorized quadrature. Re-benched 2026-09-12 (4090 box): **407 ms → 115.74 ms median @ n=2×200, d=8** (`CARGO_TARGET_DIR=/tmp/lf410 cargo bench -p katgpt-core --features linking_fold --bench bench_410_linking_fold_goat`); fold hot-path 16.25/15.76 ns @ D=8, 18.18/26.14 ns @ D=64; G1/G5 PASS. The stale pre-skip numbers (Cargo comments, module-doc table) were corrected in the filing commit.

## Tasks

### P0 — Baseline + the missing bench doc

- [ ] **T0.1** Measure `detect_linking` at n=2×1000, d=8 and record the scaling curve at n=2×{100, 200, 500, 1000} — quadratic extrapolation from 115.74 ms @ 2×200 predicts ~2.9 s, but extrapolation is not a measurement; the curve is the evidence base every P1 lever is judged against.
- [ ] **T0.2** Land the missing `.benchmarks/` doc at the then-current number. No bench doc exists for Plan 410 — the GOAT numbers live only in the Cargo feature comment and Research 391 §6. Include the 2026-09-12 re-bench numbers as the pre-Option-B baseline.

### P1 — Correctness-safe algorithmic work (zero recall loss by construction)

- [ ] **T1.1 Single-pass k-NN**: `build_epsilon_knn_graph` computes all O(n²) distances **twice** (one pass to derive ε from kth-nearest distances, a second to build adjacency) with a full O(n log n) sort per node in each. Merge into one pass: per-node bounded top-k (selection, not full sort) retained in a scratch arena, ε derived from the retained kth distances, adjacency assembled from the same retained lists. Halves the distance work at any n; removes 2× n·(n log n) sorting.
- [ ] **T1.2 Spatial cycle batching**: the pair loop is O(β_X·β_Y) `may_link` checks before any quadrature. Bucket cycles by `CycleBounds` center into a uniform grid (cell ≈ median bounding-sphere diameter); for cycle i ∈ X, only test Y cycles in nearby buckets — O(β · nearby) instead of O(β_X·β_Y). At n=2×1000 with β ≈ 2.25·n that is ~5M pairwise checks → ~β·(bucket occupancy). The skip predicate itself stays `may_link` — the grid is a prefilter, not a replacement for the provable bound.
- [ ] **T1.3 Re-gate** GOAT G1 (Hopf ±1 / unlinked 0 / fold unlinks), G2 (≤ 50 ms @ n=2×1000), G5 (determinism ×3) at the new scale. If 50 ms is still unreachable after T1.1+T1.2, record the measured floor honestly and close this issue as budget-accepted (the Option A pattern) — do not loosen the gate silently.

### P2 — Optional levers (measure recall impact first — NOT correctness-safe by default)

- [ ] **T2.1** Evaluate raising `min_cycle_len` (default 4; paper §I.1 used 30 on CIFAR-10): sweep the recall/cycle-count tradeoff on the thickened-Hopf fixture. Default change only with the recall curve recorded in the bench doc.
- [ ] **T2.2** `max_cycles_per_cloud` stays opt-in non-default (correctness-unsafe cap — the linking witness is often a LONG cycle; documented in `LinkingDetectorConfig`).

### P3 — Verdict

- [ ] **T3.1** Promote-or-keep: if ≤ 50 ms @ n=2×1000 lands AND a runtime consumer exists, propose promoting `linking_fold_detector` per the promote-if-GOAT rule (it is zero-cost-unless-invoked, so default-on is cheap — but promotion needs a consumer, not just a number). Otherwise keep opt-in audit-cadence and close.

## Follow-up tracked here (fusion idea, novelty TBD — not a task)

- [-] **Healer-surface consumer exploration**: linking detection on (rule ↔ code-shape) embedding clusters in the riir-rag span-embedding space (PCA-3D) as a retrieval top-1 confusion predictor — the paper's CIFAR-10 analogue (linking consistency ↔ confusion, Spearman r ≈ 0.48). Novelty TBD: the healer corpus is discrete `AstChunker` spans, not smooth manifolds — the k-NN-cycle machinery's manifold assumption is unverified there. Any promotion of this idea requires a PoC on the oracle-labeled retrieval fixtures (riir-clippy `retrieval_eval`) first; no plan until that PoC exists. Filed under this issue per the "fusion idea, novelty TBD → .issues entry" rule (Research 391 §6 consumer-reframe addendum).

## Provenance

- Filing commit fixes the stale-surface drift found by this session: Cargo.toml `linking_fold` umbrella + `linking_fold_detector` comments ("Issue 050 unresolved" → resolved language + fresh numbers) and the `linking_detector.rs` module-doc measured table ("wait for Option B" → what landed + Issue 757 pointer). The feature-gate-audit skill's multi-surface stale-comment class.
