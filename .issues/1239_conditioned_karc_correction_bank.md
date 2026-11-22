# Issue 1239 — Conditioned-KARC correction bank (rank-1 readout corrections keyed by condition)

**Status:** OPEN — claimed, harness in progress. Pre-registered GOAT cell below; no run scheduled. Verdict provenance: MBT (arXiv:2611.19800) research verdict package, KARC surface GAIN, reviewer AGREE final round (2026-11-22, session `42b5b2e6`) with four binding conditions folded in below.
**Claimed:** katgpt-rs-93@amo, 1795365988, expires 2026-11-24T16:46Z

## Pinned claim (§1.5 precondition)

"Conditioned-KARC — a bank of K rank-1 readout corrections over the incumbent base ridge fit, keyed by a CONDITION (σ-gated on each correction's own conformal-calibrated rolling residual), with closed-form fits (no training, no backprop) and flat state bytes at crowd scale — is a composition family member with zero shipped implementations on the KARC host (grep-verified 2026-11-22: `karc_regime_gate` = SELECT-class routing, Welford+sigmoid residual mux; `KarcBatchForecaster` = per-ENTITY stacking, never fused; `BranchBank × KARC` composition unclaimed per Bench 329's named composition-work list), for which the nearest shipped shape is an IDENTITY-keyed rank-1 deviation of the same shape class (Plan 514 Phase 3's `KarcCell.npc_directions`). This issue adjudicates select-vs-condition-vs-wide on the KARC host; it asserts no gain."

## Source + the re-keying insight (the decision content)

- arXiv:2611.19800 "Multi-Branch Transformer" — supplies the design laws only: fusion-function robustness (Table 6: parameter-free fusion within noise of learned), shared-cache-at-matched-bytes (Table 5), the W1-K2 small-scale overhead NEGATIVE (named adverse prior below), and the matched-compute cell protocol.
- **Plan 514 Phase 3** (`riir-ai/crates/riir-engine/src/karc_bridge/octree_batch.rs`) already ships the shape class keyed by IDENTITY: one shared cell-level forecaster + per-NPC `npc_directions` (unit-norm personality vectors in HLA space, read as the scalar `dot(direction, last_obs)`) — an identity-keyed rank-1 deviation, the same shape class. **Conditioned-KARC re-keys identity → condition**: one correction then serves every NPC in the same condition (crowd amortization beyond cells), or stacks with per-NPC deviations.
- Shared-cache economics (the Joint-KV leg, satisfied structurally): one delay ring, one `feature_expand`, one base `Wout`; corrections cost ~`D + d_h` floats each (~1 KB at HLA shape) vs K independent forecasters (K×8 KB) — state bytes stay flat where Plan 514 proved bytes dominate.

## Primitive

`karc_conditioned_readout` — the base ridge fit stays the incumbent; correction b is a rank-1 readout correction, fit closed-form on regime-segment data: with `u_b` frozen (authored or archetype direction), `v_b` is an ordinary ridge solve over residual targets — reuses `chunked_gram_into` + `ridge_solve` verbatim. Zero training, zero backprop; consolidation-tier work (same class as `fit_ridge`, which the runtime already runs during sleep).

```text
û = Wout_base·Ψ(x_t) + Σ_b σ(g_b(c)) · u_b · (v_bᵀ ψ_slice_b)
```

Fold order: branch index ascending, sequential — the Issue-940 determinism law.

**Condition-signal provenance (pre-registered, advisory adopted as binding here):** the conditioning variable `c` MUST come from a channel INDEPENDENT of the correction-residual σ-gate (e.g., personality blend `z` from `committed_blend`, MindState/limelight tier, combat flag, time-of-day bucket). If `c` is derived from the same rolling residual that drives `g_b`, arm A4 degenerates into a smoothed A2 and the select-vs-composition comparison measures only smoothing — that outcome is a design failure, not a result. Pin the `c` source in this file before the run.

**PINNED 2026-11-22 before any run (katgpt-rs-93@amo) — `c` source + harness:**
- **`c` = an EXOGENOUS regime label** (the in-crate stand-in for a combat flag / time-of-day bucket): the corpus switches an exogenous drive on/off in fixed blocks and the forecaster is TOLD the block label. It is computed from the schedule, never from any residual, so it is independent of the σ-gate by construction.
- **Gate deviation, disclosed:** `g_b` is an EWMA relative-gain σ-gate on correction b's own keyed residual (`σ(β·(gain − margin))`, β 30, margin 0.15, EWMA 0.05, abstain until 16 keyed obs). The full `ConformalIntervalCalibrator` is not wired; the calibration is split-conformal in SHAPE (gates warm on a calibration slice disjoint from fit and test).
- **Corpora** (double-scroll ODE of `karc_g1_dh18720.rs`, normalised to [−1, 1]; D=3, K=4 delays, M=8 Chebyshev, d_h=96; ridge λ=1e-4 for every fit): **C1 regime-switching** = double-scroll + condition-keyed constant drive on dV1 (blocks of 200 steps, c∈{0,1}); 5 seeds of observation noise (σ=0.01 normalised) = the noise-robustness arm. **C2 pure double-scroll** (c ≡ 0; corrections must abstain — G3). **C3 periodic** (3-channel multi-sine, period 20; G3 + the G5 SeasonalNaive floor reported beside it). The riir-ai NPC mixed-regime corpus is OUT (cross-repo; katgpt-rs is upstream) — it belongs to the follow-up consumer row.
- **Splits:** fit 0–60 %, gate calibration 60–70 %, test 70–100 %; one-step-ahead NRMSE; train NRMSE reported (gap check, Bench-992 lesson).
- **Arms at matched bytes** (base Wout 288 f32; 2 corrections × 99 = 198 → 486): A1 base · A2 `KarcRegimeGate` KARC-vs-SeasonalNaive (the in-crate SELECT control; `karc_regime_mux` is riir-ai) · A3 uniform mean of both corrections · A3b uniform Σ/√2 (the Bench-992 scale control) · A4 keyed σ-gated · A5 single wide fit M=13 (468 f32) · A6 two private M=7 forecasters, forecast mean (504 f32). **Disclosed extra controls:** A1c base + per-condition bias (294 f32) and A1k per-condition full readouts (576 f32) — a keyed win must beat A1c to be credited to the rank-1 bank.
- **Decision rule:** per seed r = 100·(NRMSE_arm − NRMSE_ref)/NRMSE_ref pp; δ = max(1.0, 2.5·SE); WIN r̄ < −δ, LOSS r̄ > δ. **G1 PASS** = A4 WIN vs A1 on C1 AND A5 not WIN vs A4 (A5 win ⇒ demote-to-null-record). **Attribution** = A4 vs A1c. **Promotion-bearing fixed-gate arm** = A3 WIN vs A1. **G3** = A4 not LOSS vs A1 on C2 and C3. **G2** = A4/A1 forecast-step wall ≤ 1.5× (interleaved; corpus shape + an HLA-ish shape D=8, d_h=192). **G4** = `apply_into` + `observe` alloc-free.

**Gate provenance (pinned):** `g_b` = σ-gate over correction b's OWN conformal-calibrated rolling residual (reuses the DEFAULT-ON conformal substrate; abstain-first — a never-firing correction costs only its bytes). Sigmoid, never softmax.

## Feature home

- `karc_conditioned_readout` — **inside the `estimators` umbrella in katgpt-core** (member or keyless, the `karc_regime_gate`/`epiplexity_gain` landing shape). DO NOT add a top-level feature: katgpt-core sits at **296/300** of the crates.io feature cap (`feature_cap_gate` PASS, headroom 4) and the Plan 635 Q4 count ratchet blocks umbrella-external growth.
- **No `implies karc_forecaster`** — that feature is always-on since Plan 635 Tranche 1 baked it (the dep means nothing; Cargo.toml carries the same note).
- Never joins `default`/`full` except per the Promotion ceiling below.

## Pre-registered arms

Fits are closed-form (consolidation-tier); the only training-class step is none. Corpus: the double-scroll G1 fixture (NRMSE + Lyapunov threshold-time) + a regime-switching corpus (Plan 556's) + Plan 514's mixed-regime NPC corpus (MAE).

| Arm | Composition | Role |
|---|---|---|
| A1 | incumbent single fit | floor |
| A2 | existing `karc_regime_mux` | the SELECT control — is routing already enough? |
| A3 | correction bank, fixed uniform weights (plain mean of ACTIVE corrections — Issue 1238-A6's pinned prior for independent predictors) | the unlearned-fusion default |
| A4 | σ-gated corrections on the INDEPENDENT condition `c` | **the conditioned arm** |
| A5 | single wide fit at matched total bytes (larger M) | the honest-wide control — **if it wins, the branch claim FAILS on this host** (1238-G3's shape) |
| A6 | K independent small forecasters, forecast-mean, matched bytes | the private-cache control (expect to lose on bytes AND quality — the Table-5 leg) |

## Pre-registered gates

- **G1 quality:** A3/A4 ≤ A1 on the regime-switching corpus at matched bytes; A5-win ⇒ demote-to-null-record.
- **G2 wall:** ≤ 1.5× A1 at the HLA config, per-shape pinned. **Adverse prior, named:** KARC's entire operating range (d_h 64–512) is well below MBT's smallest measured scale — MBT's W1-K2 negative (composition overhead dominating at small scale) is adverse evidence, and the mechanism that can survive is ensemble variance reduction over multicollinear basis features (scale-independent), never MBT's compute-reallocation law. The structural defense — a KARC branch is a matvec slice, not a transformer block boundary, so composition overhead largely dissolves — is a HYPOTHESIS to measure (G2), not an established fact.
- **G3 no-regression** on the pure-periodic arm (corrections must abstain where the base is already good).
- **G4 zero-alloc hot path.**
- **G5 SeasonalNaive floor preserved** on periodic segments (Report-the-Floor).
- Determinism: gated-path fold order pinned (ascending, sequential); ≥5 seeds on the noise-robustness arm. Demote losers in the same commit.

## Promotion ceiling

A G1 pass is architecture evidence whose real consumers need the runtime wiring (riir-engine `karc_bridge` `KarcCell` gains condition-keyed corrections beside `npc_directions` — follow-up row, feature `karc_conditioned_runtime` opt-in in riir-engine, unarmed boot byte-identical). The katgpt-core feature stays opt-in regardless; DEFAULT promotion requires the modelless fixed-gate arm to win its own gate, never the diagnostics alone.

## Weakest points (named at filing)

1. **The adverse scale prior is real** (see G2). A5 may simply win at matched bytes on a tiny model — the cell exists to adjudicate that; the negative is cheap and is the record.
2. **Gate provenance without training:** if the only working gate signal is the regime verdict, A4 collapses into "A2 with extra steps" and the mux wins — an honest but thin negative. That is why the `c` provenance is pinned pre-run above.
3. Mechanism class: gated residual-expert ensemble over a ridge readout is textbook; the claim is the re-keying (identity → condition) as decision-content for this stack — GOAT-tier candidate at best, Super-GOAT excluded.

## Tasks

- [x] Claim: `git fetch origin && py ../katgpt-rs/scripts/claim_lease_check.py --item 1239`; append the `**Claimed:**` line (own line, nothing after the expiry token); push BEFORE implementing (48 h expiry)
- [ ] Land `karc_conditioned_readout` inside the `estimators` umbrella in katgpt-core (NO top-level feature — cap 296/300; no `implies karc_forecaster`)
- [x] Pin the condition-signal `c` source (independent of the residual σ-gate) IN THIS FILE before any run — exogenous regime label, see PINNED block
- [ ] Implement the correction bank (closed-form fits; `chunked_gram_into` + `ridge_solve`; ascending sequential fold order)
- [ ] Harness: arms A1–A6 on the three corpora; run on amo/bmo (never m3 — job-lane law); quote `bench_preflight`'s PROVENANCE line
- [ ] Record the verdict in `.benchmarks/` (next free number); demote-to-null-record on G1 failure in the same commit as the run
- [ ] If the modelless arm wins its gate: file the promotion proposal separately (Feature Flag Discipline); never auto-promote
- [ ] Follow-up row (post-G1, separate claim): riir-engine `karc_bridge` consumer wiring

## Cross-refs

Issue 1238 (the sibling funcattn-host cell — its A6 normalization verdict is imported, never re-derived) — **closed 2026-11-22, Bench 992**: the mean prior held on TRAIN (Σ/√2 lost 6×) but INVERTED on held-out under overfit (Σ/√2 won — shrinkage-dominated metric); carry a Σ/√K scale control and a train/held-out gap check into this cell · the MBT PASS-Redirect lines (riir-train `.research/470`, riir-infer `.research/006`, riir-ai `.research/147`) · Plan 556 (`KarcBatchForecaster` — per-entity stacking, the A6 control's ancestry) · Plan 514 Phase 3 (`KarcCell`/`npc_directions` — the identity-keyed shape class) · Plan 329 (`BranchBank` — the memory-branch substrate; `BranchBank × KARC` unclaimed) · **Plan 639 (novelty-gated retention — the DRIVE-side conditioning axis, explicitly OUT of scope; do not duplicate)** · `katgpt-core::karc::regime_gate` (the SELECT control) · riir-refine `elo_k_eff` + Bench 042 (the rating-surface PASS verdict — conditioning precedent inventory).
