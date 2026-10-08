# Bench 925 — LoT-Fitted Extent Pooling as a Routing Summary Operator (Issue 925 probe)

**Date:** 2026-10-08
**Issue:** [`.issues/925_lot_fitted_pooling_routing_probe.md`](../.issues/925_lot_fitted_pooling_routing_probe.md) (now CLOSED — this record is its T4 verdict)
**Source:** [arXiv:2610.05816](https://arxiv.org/abs/2610.05816) — *Level-of-Token (LoT) Diffusion* (Nakayama et al., Oct 2026); PASS-redirects in `.research/208` + `.research/379`
**Test:** `crates/katgpt-attn/tests/issue_925_procrustes_probe.rs`
(run committed: `cargo test --release -p katgpt-attn --features pyramid_topk,hga --test issue_925_procrustes_probe -- --nocapture`;
run full: `PYRAMID_612_FULL_DIR=/Volumes/SDXC1TB/pyramid_612/full cargo test --release -p katgpt-attn --features pyramid_topk,hga --test issue_925_procrustes_probe -- --nocapture`)
**Verdict:** **NEGATIVE — the kill criterion fires.** The Procrustes-fitted summary operator (attention-centroid target + LoT Eq. 15 RMS rescale) does NOT beat plain `Mean` at equal summary dimension and equal query-time cost: **−0.0625 Recall@8 committed (sd 0.0991), −0.0098 full (sd 0.1533)** — below Mean on BOTH postures, NO-GAIN by the pre-registered rule on both. The literal key→mean fit is dead by THEOREM (identity on the data's range). The slot's operator family stays ANALYTIC-only: **no promotion, no new feature** — the fourth summary-operator negative in this slot (MSA R225/Plan 256, HGA R379/Plan 397, PISA Bench 612, now this).

## The fit (what was actually measured)

- **Candidate:** per-layer orthogonal Procrustes W over aligned pairs (block mean `μ_b`, query-marginalized attention centroid `k̄_b = Σ_j w_j k_j / Σ_j w_j`, w_j = group-averaged per-key softmax mass from the FIT partition's own queries). `M = Σ k̄_b μ_bᵀ`, one-sided Jacobi SVD (f64, deterministic), `W = U Vᵀ` with Gram-Schmidt basis completion (fit scatter is rank-deficient — see below). Summary `s_b = W μ_b`; the paper's RMS-rescale variant `s_b = W μ_b · τ/rms(W μ_b)` with fit-time `τ`.
- **Literal-alternative theorem (recorded, not measured):** pairs (k_j, μ_b) give `M_lit = C Σ_b μ_b μ_bᵀ` — symmetric PSD, so its SVD has U = V and `W_lit = I` on the scatter's range, which contains every `μ_b` the operator is ever applied to: **the literal fitted operator IS the Mean arm for any rank**. Premise verified: fit scatter σ_min/σ_max ≈ 5e-10–5e-5 (rank-deficient by construction of block means).
- **Folds (leak discipline):** leave-one-(layer, kv-head)-out — 4 folds committed, 8 full; W for layer ℓ fit ONLY from that layer's other heads' families; every family evaluated exactly once. DISCLOSED: the Issue 908 fixture is ONE prompt stream (single token-stream BLAKE3), so prompt-level disjointness is impossible on this capture; head-held-out is the strictest available split and tokens overlap across heads by construction.
- **Harness fidelity (hard-asserted comparability invariant):** this file's `mean`/`lse` arms run Bench 612's protocol verbatim and reproduce its published numbers EXACTLY — committed mean 0.7695 / lse 0.8906, full 0.6475 / 0.7793. The harness is pinned to the slot's incumbent measurement, so every delta below is measured against the same ruler as Bench 612.

## G1 — selection quality (real tensors, pooled; every family once)

Committed subset (64 non-trivial families, ~4K stride geometry, 4 folds):

| arm | recall@8 | mass-ratio | query cost per block |
|---|---|---|---|
| mean (= 612 single_mean) | **0.7695** | 0.8753 | 1 dot(d) |
| mixed_rope (shipped `MixedRopeSummarizer`) | 0.5879 | 0.8008 | 1 dot(d) |
| fit (Procrustes) | 0.7070 | 0.8364 | 1 dot(d) |
| fit_rms (Procrustes + LoT Eq. 15) | 0.7012 | 0.8352 | 1 dot(d) |
| halfvar (ceiling: per-key logits) | 0.7324 | 0.8799 | cnt dot(d) |
| lse (ceiling: exact per-key LSE) | **0.8906** | 0.9483 | cnt dot(d) |

Full set (128 non-trivial families, TRUE contiguous 4K–64K, 8 folds):

| arm | recall@8 | mass-ratio | query cost per block |
|---|---|---|---|
| mean (= 612 single_mean) | **0.6475** | 0.8030 | 1 dot(d) |
| mixed_rope | 0.4355 | 0.6859 | 1 dot(d) |
| fit | 0.6377 | 0.7957 | 1 dot(d) |
| fit_rms | 0.6455 | 0.7975 | 1 dot(d) |
| halfvar (ceiling) | 0.5439 | 0.7775 | cnt dot(d) |
| lse (ceiling) | **0.7793** | 0.8988 | cnt dot(d) |

Pre-registered verdict rule (gain iff Δ>0 AND Δ > baseline across-fold sd), computed:

| delta | committed | full |
|---|---|---|
| fit − mixed_rope | +0.1191 (sd 0.1131) → GAIN | +0.2021 (sd 0.1052) → GAIN |
| fit_rms − mixed_rope | +0.1133 (sd 0.1131) → GAIN | +0.2100 (sd 0.1052) → GAIN |
| **fit − mean (the kill row)** | **−0.0625 (sd 0.0991) → NO-GAIN** | **−0.0098 (sd 0.1533) → NO-GAIN** |

The "GAIN" rows over `mixed_rope` are real but carry no promotion weight: the shipped rival sits 18–21 pp BELOW plain Mean on real post-RoPE tensors, so beating it is beating the weakest arm in the set. Issue 925's gate names BOTH Mean and `MixedRopeSummarizer` as the equal-cost rivals that decide the kill — the fitted operator loses to Mean on both postures.

## Readings worth keeping

1. **The fit works; the signal doesn't transfer.** In-sample rotation genuinely improves alignment with the attention centroid — `mean ‖Wμ−k̄‖/‖k̄‖` vs raw: 0.383 vs 0.442 (layer 3 committed folds), 0.274 vs 0.349 (full, 13 968 pairs/fold), 0.50–0.54 vs 0.53–0.58 (layer 63). A fixed orthogonal W captures real block-mean structure — just not enough routing-relevant structure to beat the unrotated mean. The issue's LOW expectation ("at most second-moment structure; collapses into Mean and dies") is confirmed in the stronger form: it doesn't collapse INTO Mean, it lands BELOW it.
2. **RMS rescale (Eq. 15) is a no-op here** (fit vs fit_rms differ by ≤0.006 on both postures) — per-block norm equalization is not where this slot's ranking error lives; the per-key LSE ceiling (exact per-key evidence) is what moves recall (+12.1/+13.2 pp over Mean), consistent with Bench 612's ladder ordering.
3. **`mixed_rope` is a weak rival on THIS data** — a side finding consistent with its lineage (HGA R379/Plan 397 G2-proxy FAIL): the low-frequency mid-rotation rule scores post-RoPE summaries against a post-RoPE query at a different position, and the wedge correction appears to cost more than it recovers here (recall 0.588/0.436 vs Mean's 0.770/0.648). NOT actionable on this bench (its home gate is Plan 397's; the layout adaptation + zero-padded inv_freq used here is documented in the test header).
4. **halfvar loses to Mean at true lengths** (0.544 vs 0.648 full) — the order-2 Taylor rung degrades off its committed-geometry operating point, matching 612's pyramid_halfvar being the worst pyramid arm. The ceiling story stays: exact per-key LSE is the only scorer above Mean.
5. **Block-mean scatter is rank-deficient** (σ_min/σ_max 5e-10 committed → 5e-5 full): block means live on a low-dim manifold, which is exactly the premise that makes the literal key→mean fit the identity (theorem above) and limits any linear re-weighting of μ_b.

## Scope + discipline notes

- **Slot discipline held:** real Issue 908 tensors only (`778de9af8` captures, BLAKE3-verified at load); the random-key NIAH harness stayed banned; ≥3 disjoint fit/eval partitions (4/8 folds); deterministic fit (two full runs of each posture produced identical numbers); Jensen + captured-mass + orthonormality + degenerate-canary pins hard-asserted per family, all green.
- **`mixed_rope` layout adapter** (rotate-half → adjacent pairs, `inv_freq` zero-padded past rotary_dim 64): data-faithful configuration of the shipped operator, NOT rival tuning — documented in the test header with the substrate rope construction it mirrors.
- **Bonsai PQ2_0 follow-up unchanged:** still gated on the ternary whole-model capture lane (does not exist). This negative on qwen tensors LOWERS the prior for that follow-up, not the bar.
- **`pyramid_topk` stays opt-in** (Bench 612 verdict unchanged); this probe adds NO feature, NO default change — the harness is the artifact.
- Fit partitions never saw eval keys/queries (head-held-out); the single-stream fixture caveat is disclosed above and in the test header.

## Cross-references

- `.issues/925_lot_fitted_pooling_routing_probe.md` — the charter (CLOSED by this record)
- `.benchmarks/612_pyramid_goat.md` — the protocol + comparability anchor (mean/lse reproduced exactly)
- `.benchmarks/397_hga_goat.md` — the slot's first fitted-adjacent failure; the banned harness
- `.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md` — `MixedRopeSummarizer` lineage
- `.research/595_PISA_Pyramid_Sparse_Attention.md` — the scorer-ablation lane (ladder ordering confirmed)
- `.research/208_SLoD_Semantic_Level_of_Detail_KG.md` — LoT PASS-redirect (where the residue was filed)
- `crates/katgpt-attn/src/dash_attn/pyramid_topk.rs` — slot discipline + scorer ladder
- `crates/katgpt-core/src/hga/summary.rs` — the shipped rival
