# Bench 969 — Issue 969: kimi_k3 router-statistics diagnostic (Nestor App C.6) — GOAT PASS (report-only instrument)

**Status:** LANDED 2026-10-27 — K3 arm (T1+T2) measured end-to-end green; verdict hint CORPUS_INDEPENDENT_BUT_LOAD_BEARING (mixed by layer, see §Reading). T3 (reflex `pick_domain` liveness CONTROL) is the remaining half of the issue's G1.

**Date:** 2026-10-27
**Issue:** [`Issue 969`](../.issues/969_kimi_k3_router_statistics_diagnostic.md) (Research 639 / arXiv:2610.31818 Nestor App C.6)
**Instrument:** `benches/bench_969_router_stats.rs` (`required-features = ["kimi_k3_loader"]`, `harness = false`, report-only — no serving-path change, no feature promotion)

## Run posture

- amo (M5 Ultra), macOS 27.0.1, AC power; CPU-only instrument (the layer path is sequential scalar/SIMD — no GPU, no Metal), single-threaded forward ≈ 2.65 ms/token; two full stat computations 134 s.
- Weights: `riir-train/data/kimi-k3-0.40b/model.safetensors` (1.5 GB, the upstream Moonshot release — provenance recorded per the issue's T1 row; NOT a C13/C14 training artifact).
- Corpus: the reflex seated suites (ag_news / banking77 / emotion / sst5), 16 seqs each × ≤48 tokens = 64 seqs / 1503 tokens, sorted-file + row order (byte-stable).
- Gates: byte-stability (two full runs serialize identically — layers + engagement only; wall-clock meta excluded, it differs by construction — the founding run refused exactly there and was repaired, not weakened) · engagement (21,042 forced applications, 0 read-back mismatches, max shuffle rel_l2 4.8e-1 ≠ 0 — the counterfactual provably applied and reached the output; the green-zero trap pinned shut).

## The four statistics (per MoE layer, top-2 of 8)

| layer | eff choices | max load | MI bits | drift σ=.05 | drift σ=.20 | shuffle cos / relL2 / flip | random cos / relL2 / flip |
|---|---|---|---|---|---|---|---|
| 1 | 0.9234 | 0.2096 | 0.0687 | 0.0432 | 0.1424 | 0.8517 / 4.84e-1 / 0.3281 | 0.8541 / 4.84e-1 / 0.3594 |
| 2 | 0.8742 | 0.2315 | 0.0557 | 0.0665 | 0.1890 | 0.9279 / 3.30e-1 / 0.2812 | 0.9256 / 3.39e-1 / 0.2500 |
| 3 | 0.9957 | 0.1490 | 0.1336 | 0.0552 | 0.2189 | 0.9517 / 2.75e-1 / 0.2031 | 0.9494 / 2.83e-1 / 0.2188 |
| 4 | 0.4229 | 0.4378 | 0.0612 | 0.0253 | 0.0665 | 0.9949 / 2.19e-2 / 0.0000 | 0.9943 / 2.37e-2 / 0.0000 |
| 5 | 0.4975 | 0.4318 | 0.0436 | 0.0379 | 0.1091 | 0.9967 / 1.73e-2 / 0.0312 | 0.9967 / 1.81e-2 / 0.0312 |
| 6 | 0.3865 | 0.4840 | 0.0284 | 0.0319 | 0.0805 | 0.9982 / 1.08e-2 / 0.0000 | 0.9972 / 1.47e-2 / 0.0000 |
| 7 | 0.5359 | 0.4534 | 0.0520 | 0.0459 | 0.1411 | 0.9969 / 1.73e-2 / 0.0312 | 0.9970 / 1.79e-2 / 0.0312 |

Artifact: [`969_router_stats/stats.json`](969_router_stats/stats.json) (stats + meta + engagement + verdict hint).

## Reading (T4, K3 half)

- **Regime answer (single routed level):** the aggregate verdict hint is CORPUS_INDEPENDENT_BUT_LOAD_BEARING (Nestor column-level shape: MI 0.03–0.13 bits everywhere — far under the 0.28-bit expert-level mark — while the front layers' shuffle cost is large). **But the honest per-layer split is bimodal, and the record says so:** layers 1–3 are STRONGLY load-bearing (re-routing half the tokens' experts moves the final hidden 0.27–0.48 rel-L2, cos drops to 0.85–0.95, and flips the last-position argmax on 20–33% of sequences), while layers 4–7 are weakly load-bearing (rel-L2 0.01–0.02, flips ≤ 3%).
- **The layers 4–6 routers look COLLAPSED, not balanced:** effective choices 0.39–0.50 of 8 with max-load 0.44–0.48 — one expert eats ~half of every slot vote there, yet re-routing its share moves the output ~1% — consistent with a low-salience back-stack (the test-arch-quality caveat below is the right lens: an undertrained/underconverged router is exactly where this shape is expected).
- **Corpus MI is low at EVERY layer** (max 0.134 bits at layer 3) — the router's assignment is nearly corpus-independent on this 4-domain corpus; domain identity is NOT what K3's router keys on. (Nestor's expert level read 0.28 bits; our single level reads under half that.)
- **Drift under router noise is graded and monotone-ish in σ** (2.5–6.7% at σ=0.05·std, 6.7–21.9% at σ=0.20·std) — the assignments sit at nonzero margins, not at ties (an all-ties router would drift ~75% at any noise).
- **Shuffle ≈ random on every layer** (columns agree to ~1e-2): the cost is carried by the RE-ROUTING itself, not by which specific donor pair a token receives — the assignment distribution, not its fine structure, is what matters here.
- **Scope caveats carried verbatim from the issue:** K3 routes at ONE level (the paper's two-level MI split is not runnable here), and this 0.40B checkpoint is test-arch quality (owner's standing note) — this row describes THAT checkpoint's router and is never evidence about Nestor, MoE in general, or production-MoE regimes.

## G-gates (this instrument's own)

- G1 (byte-stability): PASS — two full computations serialize identically.
- G1 (engagement, K3-arm half): PASS — 0 plan mismatches over 21,042 forced applications; max shuffle rel_l2 = 4.8e-1 (the forced path provably executes).
- G1 (liveness CONTROL, reflex `pick_domain` arm): **T3 — still open** (the issue's control half; a separate seam + harness mode in riir-reflex).
- G2/G3/G4: n/a — report-only, no serving path touched, no promotion claimed. The seam additions (`MoeForwardScratch::forced_topk`, `KimiK3Runtime::forced_router_plan`) are `None` on every production path (one predictable branch), and the whole-model default clippy surface is clean at both postures (default + `kimi_k3_loader`).

## What landed (this bench's diff)

- `crates/katgpt-transformer/src/moe.rs` — `forced_topk` scratch slot + step 4.5 override (weights recompute from the token's own sigmoid scores over the forced indices — assignment-only counterfactual).
- `src/kimi_k3/model.rs` — `forced_router_plan` runtime field + `arm_forced_router` called in BOTH forward loops (production + traced) so the slot is refreshed/cleared on every layer visit (stale state cannot leak across tokens); `reset()` disarms.
- `benches/bench_969_router_stats.rs` + the `[[bench]]` manifest row.

## Follow-ups

- T3 (reflex control arm) — the issue's remaining half.
- The layer 4–6 collapse shape is worth one follow-up question on a REAL checkpoint (not this test-arch one) before any generalization is attempted — file only if a consumer exists (the issue's standing bar for the 0.40B artifact).
