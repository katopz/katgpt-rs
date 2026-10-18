# Issue 948 — Terminal-RSI modelless extractions: family-calibrated partial-credit grader + success-offset lexicographic ordering

**Status:** OPEN — fusion-idea intake from riir-train Research 477 (arXiv:2610.22192 Terminal-RSI, 2026-10-18 panel). Closed-world scoring math (no training, no domain semantics); novelty TBD at the GOAT level per item. Consumers verified at the architecture level; no auto-promotion.

## T1 — FDR family-calibrated partial-credit grader (STRONGEST)

The paper's Family Dense Reward `R = max(P/D_f, 0)` is fully offline-computable: count per-trajectory check passes `P`, freeze `D_f` = per-family 90th percentile of `P` on a reference corpus, score. The paper independently converged on the lesson our own harness paid for ("fused-gate birth thresholds do NOT transfer: fit per suite") — frozen per-family normalization, never per-batch.

- **What ships:** the hinge — `delta_absorb.rs::observe_hint_delta` clamps `delta.value.max(0.0)`.
- **The gap:** raw unnormalized values; no family divisor. HintDelta already carries a natural family key (template/domain).
- **⚠ Freeze gate (mandatory):** the 90th percentile is the exact statistic the percentile-index audit exists for — `sorted[(n·0.9) as usize]` lands on n−1 (the MAX) for n ≤ 10 and reports one observation under a percentile's name. The freeze path must assert **tail support** (`n_f − idx ≥ 10`, i.e. family reference corpora ≥ ~100 rows) and REFUSE to freeze `D_f` below it — loud, never a silent thin-corpus divisor.
- **Consumers:** cgsp hint-delta absorb normalization · teacher-data grading (riir-train B6) · corpus filtering (deterministic, freezable) · harness board cross-suite partial credit.
- **GOAT shape:** G1 ordering gate (FDR ordering beats raw-P ordering vs adjudicated quality on a held-out slice) + G1 stability (bootstrap-resampled D_f variance under a pinned bound) + G4 zero-alloc scoring. Not a gate today — intake first.

## T2 — SPR success-offset lexicographic ordering

The paper's Success-Prioritized Reward `R = S + P/(P+D'_f)` encodes a pure ordering law: the strict class dominates — **no amount of partial credit ever promotes a failure over a success**. The soft term is a saturating monotone squash (Michaelis-Menten = sigmoid-family in log space; house sigmoid-not-softmax rule compatible).

- **Consumers:** every candidate-ranking site that consumes scalar scores where a verified-pass and a near-miss differ by a small delta — DDTree/MCTS child stats, drafter candidate ordering, CGSP replay/absorb selection, corpus filter admission. Sort by `(strict_class, soft_score)` lexicographically.
- **GOAT shape:** by-construction property test (separation is executable, not measured) + paired read on the frozen candidate corpus — if lexicographic selection is byte-identical to incumbent selection on today's data, record the honest no-op and keep the invariant as a drift guard.
- **Non-gating:** reproducing the paper's 50.09%→28.72% failure-reward-mass figure is training-track business (Plan 464); the modelless residue is the offline separation diagnostic `mass_on_failures` as an eval-board metric (R477 F1).

## T3 — Decay-free bounded-credit reshape (minor; RE-SCOPED at verdict round 1)

**Incumbent record (the panel's original premise was stale — caught by the verdict reviewer, verified in source):** the ~200-cycle saturation of `PriorityTableBandit` is the PRE-Issue-303 additive-absorb behavior (decay = 1.0). The mitigation SHIPS: decayed absorb `p ← p·decay + reward`, capped at 1.0, with `DEFAULT_BANDIT_DECAY = 0.7` (`riir-engine/src/cgsp_runtime/runtime.rs` L69, Issue 303) applied by every construction path (`uniform()`/`from_seed()`); `with_decay` exists only for backward-looking benchmark baselines. The tuning record shows 0.99 and 0.9 were measured and rejected ("with decay = 0.99, any arm with reward > 0.01 saturates"; steady-state analysis: an arm at constant reward r settles at `r/(1−DECAY) ≈ 3.3·r` at 0.7). Issue-303 acceptance = variance > 1e-4 after 1000 cycles. The `cross_game_bench.rs` saturation doc this intake originally cited (L54–58, L337–343, `t53_info_bandit_saturation`) describes the decay=1.0 posture and is STALE against the default.

- **Reframed differential claim:** a saturating reshape `P/(P+D')` is a **decay-free alternative** — bounded, comparable credit mass WITHOUT exponentially discarding old evidence (decay forgets; the reshape saturates but keeps). Not a saturation fix — that defect is closed.
- **GOAT (pinned posture):** reshape-vs-decay comparison AT the shipped 0.7 default (or an explicit decay=1.0 baseline with the bench-doc staleness named) on the frozen episode set: post-shift regret retention (does keeping old evidence beat forgetting it?) + G3 bit-identity with shaping off. A bench run against the labeled decay=1.0 baseline posture alone would be vacuous.
- **Pre-existing rot this intake depends on (named so the next reader doesn't re-derive the false premise):** `runtime.rs` doc comments at L74, L87, L122 still say "0.99 (default)" while the const is 0.7; `dual_pool_bridge.rs` L1001–1002 likewise says "0.99 default — Issue 303"; `cross_game_bench.rs` L54–58 + L337–343 document saturation as current behavior. Doc-only repair candidates in riir-ai (out of scope for this issue).

## Provenance

Distilled from riir-train `.research/477_Terminal_RSI_Recursive_Terminal_Agent_Recipe.md` (three-track panel ran: No-GD advocate extraction items 4/5/6; discards + signal-diffs recorded THERE; T3 re-scoped at verdict round 1 — its original saturation premise was the pre-Issue-303 behavior). Training-side half of the same mechanisms: riir-train Plan 464. External prior art is the paper itself (FDR/SPR are its contribution; the class is classical reward shaping) — internal novelty only, which caps this at GOAT-tier intake, not Super-GOAT.
