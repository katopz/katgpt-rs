# Issue 940: Float reduction-order + tie-band determinism audit (handwired distill, Research 621)

**Status:** Open — audit, no code change implied yet
**Source:** [Research 621](../.research/621_Handwired_Exact_Differentiable_Program_Circuits.md) — chenglou/handwired @ `4ea7fcb2`, `docs/patterns.md` §"Why a tree gives the loop's answer" + `emitArgmax` tie-band law.

## Why

Two measured determinism hazards from handwired apply directly to any value that is (a) an **f32 reduction** or (b) a **near-tie argmax pick**, wherever the result crosses the sync boundary or feeds committed/replayed state:

1. **Reduction-order grouping.** FP32 addition is not associative; loop order vs tree/parallel order gives different bits (their example: loop 799.300048828125 vs pairwise 799.2999877929688). Exact in any grouping only for integers < 2^24 and power-of-two-grid sums while totals stay small. Our contract: "Deterministic replay, quorum sync, and anti-cheat require bit-identical reconstruction across nodes" — any reduction whose order can vary (SIMD lane reduction, rayon, GPU) between the writer and the replayer/quorum node breaks it.
2. **Argmax tie band.** A tournament/tree argmax with float-tolerance comparisons (~9e-8, up to 2.4e-7 for some scores) is non-transitive and can pick a different candidate than the sequential loop. Exact repair = compare-based combine (`a > b ? a : b` semantics pinned to one operand on ties).

Known synced f32 aggregate by contract: the flock centroid — per the global latent-vs-raw "What Goes Where" rules (workspace AGENTS.md), not riir-ai AGENTS.md: "Flock centroid → aggregated latent-ish (average position). Synced as-is because it's a summary, not per-entity truth." That audit half belongs to the game side — filed as riir-ai Issue 1074; this issue keeps the katgpt-rs scope (below). katgpt-rs-side scope: selector/pruner picks and any belief/kernel f32 sums that feed committed state.

## Tasks

- [ ] **T1** Enumerate every katgpt-rs-side f32 reduction whose result is committed, exported, or replay-compared (belief/kernel sums, selector aggregates). For each: is the reduction order a single fixed sequential order, or platform/lane dependent? (The game-side sync-surface half — flock centroid etc. — is riir-ai Issue 1074, not here.)
- [ ] **T2** For each site that is order-dependent: either pin the order (documented sequential fold) or move the quantity to exact arithmetic (integer/grid), or reclassify it as local-only (never compared across nodes) with the decision recorded in the site's docs.
- [ ] **T3** Audit near-tie argmax/tournament picks that feed decisions (goal salience, limelight picks, pruner selection): confirm tie-breaking is a single fixed rule (first-index / stable), not tolerance-based; add the tie-band fact to the selection module docs.
- [ ] **T4** Write the law once, in the right home: "an f32 reduction or pick that crosses the sync/quorum boundary pins its reduction order and tie rule; anything else must be reclassified as local." Reference Research 621 + handwired patterns.md.

## References

- handwired @ `4ea7fcb2` — patterns.md (grouping analysis, precision inversion: tree grouping can be *more* accurate — 0.26 px vs 5.67 px vs float64 at 10k rows), expressions.md (`emitHighestScoreExact`).
- riir-ai AGENTS.md sync-boundary law; katgpt-rs latent-vs-raw rules.
