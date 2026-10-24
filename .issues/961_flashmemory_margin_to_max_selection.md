# Issue 961 — FlashMemory selector: row-relative margin-to-max selection (`flashmemory_margin_select`)

**Status:** Open — **T1 DONE 2026-10-25** (`c6fb48962`: opt-in primitive + 4 tests); T2 (forward-level G1) and T3 (promotion verdict) open.
**Claimed:** katgpt-rs-m5-452e1, 1792865680, expires 2026-10-27T12:00:00+00:00
**Origin:** riir-train [Bench 669](../../riir-train/.benchmarks/669_issue452_e2_modelless_vs_trained_bonsai2_ladder.md) (riir-train Issue 452 E2) — an offline selector evaluation on the Bonsai-2 PQ2_0 8K/32K/64K/128K ladder.

## Finding that motivates it

`GqaFlashMemorySelector` / `FlashMemorySelector` select a block iff `σ(s_b) ≥ threshold` with `s_b = q·c_b·attn_scale` — an ABSOLUTE rule, so a per-row logit offset (which the softmax the selector approximates is invariant to) moves the selection. A ROW-RELATIVE rule on the same scores — select iff `s_b ≥ max_b' s_b' − δ`, i.e. `σ(s_b − s_max + δ) ≥ 0.5` — measured, at recall 0.97 against the cum-90% golden set (riir-train Bench 669, selection rate, lower is better):

| rung | absolute | **margin** | trained d5 (native) | oracle |
|---|---|---|---|---|
| 8K | 0.892 | 0.844 | 0.783 | 0.262 |
| 32K | 0.776 | **0.709** | 0.765 | 0.115 |
| 64K | 0.841 | **0.715** | 0.832 | 0.099 |
| 128K | 0.720 | **≈0.68** | 0.696 | 0.083 |

The margin rule dominates the shipped rule at every rung and matches/beats a native-trained indexer from 32K up — a modelless gain. Its operating δ also drifts smoothly with context (6.3 → 8.4) where the absolute threshold needed jumps (0.30 / 0.25 / 0.15 / 0.30).

Cost: the scores are already in `scores_buf`; the rule adds one max-fold per head per refresh — zero allocation.

## Tasks

- [x] **T1 — opt-in primitive.** DONE `c6fb48962`. Feature `flashmemory_margin_select` (implies `flashmemory_sparse`): `with_margin(delta)` builder on both selectors; one shared selection helper used by both (`threshold` path byte-identical when no margin is set). Unit tests: margin semantics, offset invariance, threshold-path parity, empty-row edge.
- [ ] **T2 — forward-level G1.** Offline recall/sel is a proxy. Measure attention-OUTPUT fidelity (riir-train `plan545_precompute_bonsai_gpu --d1-*` path, cos vs dense) at MATCHED retention, margin vs absolute, on ≥2 rungs. Needs a riir-train-side `--d1-modelless-margin` arm (that repo) and the M5 GPU.
- [ ] **T3 — GOAT verdict.** G1 = T2; G2 = selection latency at n_blocks ∈ {512, 2048, 4096} (`--release`), margin ≤ 1.1× absolute; G3 = default-off build unchanged; G4 = alloc-free (shared helper). Promote to default only if G1 holds — and even then the shipped `threshold` default stays the default RULE unless a consumer opts in (the operating δ is context-dependent).

## Non-goals

E1's ≥90% KV reduction: Bench 669 shows no selector over the block-mean key centroid gets near the oracle (best ≈68% retention at 128K vs 8.3% oracle). That is a ranking-signal problem, not this issue's.
