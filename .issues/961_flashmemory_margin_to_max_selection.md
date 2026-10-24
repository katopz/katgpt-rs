# Issue 961 — FlashMemory selector: row-relative margin-to-max selection (`flashmemory_margin_select`)

**Status:** Open — **T1 DONE 2026-10-25** (`c6fb48962`: opt-in primitive + 4 tests); T2 (forward-level G1) and T3 (promotion verdict) open.
**Claimed:** katgpt-rs-m5-452e1, 1792865680, expires 2026-10-27T12:00:00+00:00
**Origin:** riir-train [Bench 669](../../riir-train/.benchmarks/669_issue452_e2_modelless_vs_trained_bonsai2_ladder.md) (riir-train Issue 452 E2) — an offline selector evaluation on the Bonsai-2 PQ2_0 8K/32K/64K/128K ladder.

## Finding that motivates it

`GqaFlashMemorySelector` / `FlashMemorySelector` select a block iff `σ(s_b) ≥ threshold` with `s_b = q·c_b·attn_scale` — an ABSOLUTE rule, so a per-row logit offset (which the softmax the selector approximates is invariant to) moves the selection. A ROW-RELATIVE rule on the same scores — select iff `s_b ≥ max_b' s_b' − δ`, i.e. `σ(s_b − s_max + δ) ≥ 0.5` — measured at recall 0.97 against the cum-90% golden set on each query's CAUSAL prefix, held-out query positions, seeds 42 / 7 (riir-train Bench 669, selection rate, lower is better):

| rung | absolute | **margin** | trained d5 (held-out) | oracle |
|---|---|---|---|---|
| 8K | 0.892 / 0.863 | **0.838 / 0.801** | 0.860 / 0.851 | 0.370 / 0.341 |
| 32K | 0.760 / 0.768 | **0.677 / 0.719** | 0.708 / 0.729 | 0.159 / 0.167 |
| 64K | 0.818 / 0.821 | **0.680 / 0.702** | 0.855 / 0.840 | 0.127 / 0.131 |
| 128K | 0.677 / 0.622 | **0.627** / 0.572 | 0.672 / **0.569** | 0.114 / 0.097 |

The margin rule beats the shipped rule in 8/8 cells and a held-out native-trained indexer in 7/8 (one tie) — a modelless gain. Its operating δ drifts smoothly with context (≈6.2 → 8.3) where the absolute threshold needed jumps (0.30 → 0.24 → 0.14 → 0.29). (Bench 669's first tables, quoted in the first version of this issue, were whole-row and row-leaky; the margin-vs-absolute direction survived the repair.)

Cost: the scores are already in `scores_buf`; the rule adds one max-fold per head per refresh — zero allocation.

## Tasks

- [x] **T1 — opt-in primitive.** DONE `c6fb48962`. Feature `flashmemory_margin_select` (implies `flashmemory_sparse`): `with_margin(delta)` builder on both selectors; one shared selection helper used by both (`threshold` path byte-identical when no margin is set). Unit tests: margin semantics, offset invariance, threshold-path parity, negative-margin refusal.
- [ ] **T2 — forward-level G1.** Offline recall/sel is a proxy. Measure attention-OUTPUT fidelity (riir-train `plan545_precompute_bonsai_gpu --d1-*` path, cos vs dense) at MATCHED retention, margin vs absolute, on ≥2 rungs. The riir-train arm exists: `plan545_precompute_bonsai_gpu --d1-modelless-margin <δ>` (riir-train `19975145`); needs the M5 GPU and a Bonsai-2 d5 ckpt for the trained slot.
- [ ] **T3 — GOAT verdict.** G1 = T2; G2 = selection latency at n_blocks ∈ {512, 2048, 4096} (`--release`), margin ≤ 1.1× absolute; G3 = default-off build unchanged; G4 = alloc-free (shared helper). Promote to default only if G1 holds — and even then the shipped `threshold` default stays the default RULE unless a consumer opts in (the operating δ is context-dependent).

## Non-goals

E1's ≥90% KV reduction: Bench 669 shows no selector over the block-mean key centroid gets near the oracle (best ≈68% retention at 128K vs 8.3% oracle). That is a ranking-signal problem, not this issue's.
