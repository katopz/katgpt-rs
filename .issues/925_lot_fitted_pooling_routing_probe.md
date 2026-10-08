# Issue 925: LoT-fitted extent pooling as a routing summary operator — probe with MixedRope/Mean baselines

**Status:** CLOSED 2026-10-08 — documented NEGATIVE (kill criterion fired). Bench record: `.benchmarks/925_lot_fitted_pooling_probe.md` (fitted − mean = −0.0625 committed / −0.0098 full, NO-GAIN on both postures; literal key→mean fit is identity BY THEOREM; harness reproduces Bench 612's mean/lse numbers EXACTLY). Fourth summary-operator negative in this slot; no feature, no promotion; `pyramid_topk` stays opt-in. Reopen trigger: Bonsai PQ2_0 capture lane + a non-linear fit target. Prior rated LOW — confirmed.

## Source

arXiv:2610.05816 "Level-of-Token (LoT) Diffusion" (Nakayama et al., Oct 2026) — PASS verdict (see the PASS-Redirects lines in `.research/208_SLoD_Semantic_Level_of_Detail_KG.md` and `.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md`). The paper's fitted extent-pooling stack (orthogonal Procrustes bases, Eq. 14; RMS rescale, Eq. 15) is the one modelless extraction whose mechanism-grade residue survived the adversarial panel + verdict negotiation; everything else is either shipped substrate or published prior art.

## What the probe is

A **fitted static, query-independent** summary operator for the sparse-attention routing slot: offline Procrustes fit (SVD on aligned pairs of real K tensors → their pooled summary), one matvec at query time, equal summary dimension and equal query-time FLOPs vs the shipped analytic summaries. The paper's own ablations (multilevel > binary; analytic-lift > direct-reinterpretation) say merge-operator quality is load-bearing at compression, and this slot's operator family has only been explored analytically (Mean / MeanPlusHalfVar / ExactLse). A data-FITTED operator is untried.

The honest contrast (verified `crates/katgpt-attn/src/dash_attn/pyramid_topk.rs:465-476`): `MeanPlusHalfVar` computes mean + ½·Var over the ACTUAL per-query child logits — query-dependent, no isotropic assumption, strictly more information per query but more FLOPs. A Procrustes summary is precomputed and query-independent; its only edge is zero query-time cost beyond the matvec. Expectation: LOW — a linear operator + RMS rescale carries at most second-moment structure; if the real-tensor key distributions are near-isotropic the candidate collapses into Mean and dies.

## Slot discipline (binding)

- The random-key NIAH harness is **BANNED** for this slot (`crates/katgpt-attn/src/dash_attn/pyramid_topk.rs` header — "every selection-quality claim must replay REAL pretrained checkpoint tensors"). A Procrustes basis fitted on random keys has no structure to learn — the HGA G2-proxy FAIL root cause (`.benchmarks/397_hga_goat.md`: "group summaries (mixed-RoPE mean-pool of 16 random keys) dilute the single-needle signal") is one a fitted basis cannot fix by construction. This issue does NOT claim to fix that failure.
- **Tensor source: Issue 908 (closed, `778de9af8`) qwen 27B Q4_K_M FA-layer captures** — the same set Bench 612 used (`.benchmarks/612_pyramid_goat.md` gate setup: committed subset 16 K bins + 30 Q mats runs in CI; full 1.2 GB set env-gated, BLAKE3-verified). Committed subset = CI arm; full set = gate run. Direct comparability with the Bench 612 numbers is required.
- Bonsai PQ2_0 tensors are FOLLOW-UP WORK gated on the ternary whole-model capture lane existing (`riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` is the owner-preferred model; the capture lane does not exist yet). Not a deferral — follow-up.
- **Leak ban:** fit and eval must sit on DISJOINT prompt partitions. The fit must never see the eval split's queries or keys.

## Gate

1. **Metric:** Recall@k of the true top block on REAL tensors at EQUAL summary dimension and EQUAL query-time cost — routing ranking, never LoT's reconstruction/noise-preservation objective (irrelevant to routing).
2. **Equal-cost rivals (decide kill here):** `MixedRopeSummarizer` (`crates/katgpt-core/src/hga/summary.rs`) and plain `Mean`.
3. **Quality ceiling (informs, never decides):** `MeanPlusHalfVar` + `ExactLse`, with their extra query-time cost stated beside every number (PISA-2/PISA need child logits per query; the fitted summary does not).
4. **Spread:** captured tensors + offline fit are deterministic — seed variance is meaningless. Use **≥3 disjoint fit/eval prompt partitions**; the kill margin is measured against the ACROSS-PARTITION spread of the baseline.
5. **Kill criterion:** no Recall@k gain over `MixedRopeSummarizer` at equal cost ⇒ close as a documented negative in the slot's ledger (Bench-style `.benchmarks/` record, same commit discipline as Bench 612).

## Tasks

- [x] T1: fit harness — Procrustes SVD on aligned (pooled-group, summary) pairs from the Issue 908 (`778de9af8`) committed-subset captures; offline, no GD. DONE: `crates/katgpt-attn/tests/issue_925_procrustes_probe.rs` — per-layer orthogonal Procrustes (one-sided Jacobi SVD, f64, deterministic; Gram-Schmidt basis completion — scatter is rank-deficient, σmin/σmax 5e-10–5e-5); target = query-marginalized attention centroid k̄_b from the FIT partition's own queries; LoT Eq. 15 RMS rescale as the second variant.
- [x] T2: replay arm — Recall@k of fitted summary vs `MixedRopeSummarizer` + `Mean` at equal dim/cost, ≥3 disjoint partitions. DONE: leave-one-(layer,head)-out — 4 folds committed / 8 full, every family evaluated exactly once; fit − mean = **−0.0625 / −0.0098 → NO-GAIN both postures** (fit − mixed_rope = +0.119/+0.202, but mixed_rope sits 18–21 pp below plain mean — beating it carries no promotion weight).
- [x] T3: ceiling rows — `MeanPlusHalfVar` + `ExactLse` numbers with query-time cost stated, from the same partitions. DONE: halfvar 0.7324/0.5439 (loses to mean at true lengths), lse 0.8906/0.7793 (+12.1/+13.2 pp over mean — the only arm above mean); per-block cost column in the bench tables.
- [x] T4: verdict record — promote-shaped GOAT table or documented negative, cross-referenced here; close the issue either way. DONE: `.benchmarks/925_lot_fitted_pooling_probe.md` (documented negative; catalog entry #44 in `.docs/09_feature_catalog/negative_results.md`).

## Cross-references

- `.benchmarks/925_lot_fitted_pooling_probe.md` — THE VERDICT RECORD (documented negative)
- `.docs/09_feature_catalog/negative_results.md` #44 — catalog entry
- `.benchmarks/397_hga_goat.md` — HGA G2-proxy FAIL + root cause (random-key dilution; banned harness)
- `.benchmarks/612_pyramid_goat.md` — PISA iso-quality FAIL (−0.133 Recall@8 @32K+); tensor capture provenance; the comparability anchor this probe reproduces exactly
- `.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md` — HGA note (the slot's incumbent summary)
- `.research/595_PISA_Pyramid_Sparse_Attention.md` — PISA note (the operator-ablation lane)
- `.research/208_SLoD_Semantic_Level_of_Detail_KG.md` — the LoT PASS-redirect that filed this residue
- `crates/katgpt-attn/tests/issue_925_procrustes_probe.rs` — the probe harness (the artifact)
- `crates/katgpt-attn/src/dash_attn/pyramid_topk.rs` — slot discipline + `PyramidScoreMode` arms
- `crates/katgpt-core/src/hga/summary.rs` — `MixedRopeSummarizer`
- Issue 908 (closed, `778de9af8`) — the qwen 27B Q4_K_M FA-layer tensor captures
