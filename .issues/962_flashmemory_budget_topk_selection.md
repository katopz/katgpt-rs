# Issue 962 — FlashMemory selector: budget (top-ρ) block selection (`flashmemory_budget_select`)

**Status:** Open — T1 DONE (opt-in primitive + 5 tests, this commit); T2 (128K forward-level confirmation) running; T3 (riir-infer serving consumer) open.
**Claimed:** katgpt-rs-m5-452e1, 1792865680, expires 2026-10-27T12:00:00+00:00
**Origin:** riir-train Issue 452 E1 — after the margin rule (Issue 961, closed) failed its forward-level G1, the D1 top-ρ harness (riir-train `cf8144e1`, `682d9eae`) compared selectors at EXACTLY matched retention on attention-output cos.

## Finding that motivates it

Bonsai-2 PQ2_0, 32K, 128 queries × layers 19/35/51, one forward (riir-train `.raw/i961/e1_topk_32k_v2.log`). Each arm keeps the top `round(ρ·n) − 1` complete causal blocks by its own score plus the current block — equal retention by construction. cos(sparse, dense attention output), mean per layer 19 / 35 / 51:

| ρ (≈ sel) | **centroid top-ρ** | Quest min/max bound | window | sink+window |
|---|---|---|---|---|
| 0.05 | **0.939 / 0.956 / 0.974** | 0.880 / 0.852 / 0.784 | 0.881 / 0.859 / 0.913 | 0.883 / 0.882 / 0.933 |
| 0.10 | **0.967 / 0.975 / 0.986** | 0.928 / 0.899 / 0.879 | 0.903 / 0.886 / 0.936 | 0.912 / 0.923 / 0.958 |
| 0.15 | **0.979 / 0.983 / 0.991** | 0.951 / 0.924 / 0.920 | 0.918 / 0.902 / 0.948 | 0.926 / 0.943 / 0.968 |
| 0.30 | **0.992 / 0.995 / 0.997** | 0.979 / 0.962 / 0.965 | 0.951 / 0.926 / 0.966 | 0.957 / 0.966 / 0.980 |

The centroid-dot score (the shipped one) beats Quest bounds and both recency baselines at every ρ and layer, worst-case cos included. And the BUDGET rule beats the shipped ABSOLUTE rule on the same score at lower retention (layer 35: top-ρ 0.30 → 0.9951 / 0.9148 at sel 0.298 vs thr 0.5 → 0.9888 / 0.8763 at sel 0.413; layer 19: 0.9923 / 0.985 at 0.298 vs 0.9881 / 0.945 at 0.370). The absolute threshold wastes retention because one threshold meets per-layer, per-row logit scales; a fixed fraction per row does not.

## Tasks

- [x] **T1 — opt-in primitive.** `with_budget(rho)` on `GqaFlashMemorySelector` and `FlashMemorySelector` behind `flashmemory_budget_select`; the selectors' rule is now one private `RowRule` enum (Absolute / Margin / Budget) through the shared `select_scored_row`; top-k via `select_nth_unstable_by` on a reserved rank scratch (zero alloc in steady state, gated by a counting-allocator test); deterministic ties (lower index); output ascending. Tests: `crates/katgpt-attn/tests/flashmemory_budget_select.rs` (5).
- [ ] **T2 — 128K confirmation** (the oracle's ~10% regime): same harness, 256 queries (riir-train `.raw/i961/e1_topk_128k.sh`, running).
- [ ] **T3 — serving consumer.** riir-infer `forward_attention_layer_flashmemory` builds the selector from `FlashMemoryConfig`; wiring `with_budget` there (and an end-to-end quality gate — next-token agreement / NLL vs dense at ρ ≈ 0.1–0.15) is a riir-infer issue to file once T2 holds.

## Promotion note

`flashmemory_sparse` itself is opt-in, so "default" here means the selector's default RULE. The absolute rule stays the default until T3's end-to-end gate; the D1 table is the forward-level G1 evidence.
