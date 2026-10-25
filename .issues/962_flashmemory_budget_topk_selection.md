# Issue 962 — FlashMemory selector: budget (top-ρ) block selection (`flashmemory_budget_select`)

**Status:** Open — T1 DONE (`aa439587c`); **T2 DONE** (128K confirms, riir-train Bench 671); T3 (serving consumer + end-to-end gate) filed as riir-infer Issue 1117.
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

- [x] **T1 — opt-in primitive.** DONE `aa439587c`. `with_budget(rho)` on `GqaFlashMemorySelector` and `FlashMemorySelector` behind `flashmemory_budget_select`; the selectors' rule is now one private `RowRule` enum (Absolute / Margin / Budget) through the shared `select_scored_row`; top-k via `select_nth_unstable_by` on a reserved rank scratch (zero alloc in steady state, gated by a counting-allocator test); deterministic ties (lower index); output ascending. Tests: `crates/katgpt-attn/tests/flashmemory_budget_select.rs` (5).
- [x] **T2 — 128K confirmation** (the oracle's ~10% regime, 256 queries): HOLDS and strengthens — centroid top-ρ beats Quest, window and sink+window at every ρ and layer; ρ 0.10 keeps mean cos 0.980 / 0.990 / 0.991 (min 0.957 / 0.850 / 0.917). Open risk: layer 35's worst case stays ~0.85 for ρ ≤ 0.15 (riir-train Bench 671 reading 5).
- [x] **T3 — serving consumer + end-to-end gate:** riir-infer Issue 1117 — LANDED 2026-10-25 (zed-m3-continue): the ternary decode-lane wiring `51f52f4` + the incremental-substrate method `3d3152fbd` (this repo) + the G1 battery (ρ=1.0 bit-identical to dense over 700 synth steps, incremental ≡ rebuild, budget moves+prunes); the end-to-end teacher-forced NLL/top-1 gate (T3/T4 of 1117) runs on the real Bonsai-2 PQ2_0 at ctx 32K, ρ ∈ {0.05,0.10,0.15,0.30} — the riir-infer issue carries the record.

## Promotion note

`flashmemory_sparse` itself is opt-in, so "default" here means the selector's default RULE. The absolute rule stays the default until T3's end-to-end gate; the D1 table is the forward-level G1 evidence.
