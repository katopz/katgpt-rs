# Plan 628: Fathom extraction substrate — Phase 0 (upstream-first)

**Date:** 2026-10-20
**Research:** [katgpt-rs/.research/627_Cloudflare_Clairic_Extraction_Models.md](../.research/627_Cloudflare_Clairic_Extraction_Models.md)
**Proposal:** [katgpt-rs/.proposals/016_fathom_extraction_stack.md](../.proposals/016_fathom_extraction_stack.md) (family charter; Phases 1+ are owner-gated there)
**Source:** Cloudflare Clairic preview (2026-10-19, hosted-only, open weights pending) — the extraction-model class
**Target:** `katgpt-rs/crates/katgpt-core/src/extraction_wire/` (contract only) + NEW member crate `katgpt-rs/crates/katgpt-extract/` (everything that grows: parsers, span scorer, estimate/forecast heads, Temper substrate) + features `extraction_wire`, `span_pointer`, `extraction_estimate`
**Status:** Active — Phase 0 (substrate; NOT owner-gated; the reflex precedent: `Lz4FlexDrafter::score_into` landed before riir-reflex existed). Claude verdict round 1: REVISE addressed 2026-10-20 — member-crate placement, Duckling/Recognizers-Text grammar source, estimate/forecast conditioner split, external G5 fixture.

---

## Goal

Land the open, weightless substrate of the Fathom extraction stack upstream: the `extraction_wire` contract (one-definition, `decision_wire` R7-freeze shape) in katgpt-core, and a new member crate `katgpt-extract` holding everything that GROWS with grammar/domain — the `span_pointer` span-answer primitive, deterministic typed parsers (grammar design consumed from Meta Duckling + Microsoft Recognizers-Text), the cross-sectional estimate conditioner and time-series forecast head over Plan 340, and the Temper gate substrate. Each feature-flagged, benched, GOAT-gated. When (owner-GO) `riir-fathom` is born, it assembles these; nothing in Phase 0 depends on that birth.

**Crate split rationale (verdict round 1):** the wire types are a pure contract — they belong in katgpt-core beside `decision_wire`. Parsers and the span scorer grow with date/locale grammar and alias tables — they do NOT belong in the core leaf; a member crate keeps the leaf clean while staying in-workspace. The estimate head consumes katgpt-core's conformal substrate via a normal dep edge.

## Phase 0 — Substrate (CORE)

### Tasks

- [ ] **T0.0** run the `boundary-guard` skill BEFORE any module/crate creation (new member crate `katgpt-extract` + its dep edge into katgpt-core) — record the verdict in this plan; BOUNDARY.md rows if the contract requires them.
- [ ] **T0.1** `extraction_wire` module in katgpt-core — field-spec vocabulary (`text|number|date|time|datetime|duration|list|estimate|forecast` + constraints + instructions + reference_date), answer envelope (value + span + `p_null` + `confidence`; median/p10/p90/p_below/p_above for estimates; `guess` on null), `contract_version` + additive-only law + unknown-field refusal. Serde-only, zero-dep, wasm-safe.
- [ ] **T0.2** member crate `katgpt-extract`: `span_pointer` scorer (feature `span_pointer`) — candidate-window scoring: BM25/ngram lexical + hashed-feature embedding similarity + optional alias tables; sigmoid normalization; exact input-slice offsets. Zero-alloc API (`score_into(&mut [SpanScore])`). Deps: katgpt-core only.
- [ ] **T0.3** typed parsers in `katgpt-extract` — number (integer/decimals/unit), date/time/datetime (ISO + reference_date anchoring), duration (unit), list-of-typed (input order, spans). **Grammar design consumed from Meta Duckling (Apache-2.0) + Microsoft Recognizers-Text (MIT) — translate their dimension/grammar shapes, credit in module docs; never reinvent.** Grammar-first, span-fallback; constraints enforced by construction.
- [ ] **T0.4** estimate + forecast heads in `katgpt-extract` (feature `extraction_estimate`):
  - **Estimate (cross-sectional):** comparables corpus = (a) tabular/structured content parsed FROM the state (Clairic's own home-price example carries the comps in the request), (b) the per-deployment outcomes ledger as it accumulates (Phase 2 — ledger interface stubbed now). kNN over comparables → point predictor → `ConformalIntervalCalibrator` overlay → median/p10/p90 on the caller's grid (linear/log).
  - **Forecast (time-series):** `SeasonalPoolForecaster`/KARC family + the same calibrator in h-step residual mode, per-step intervals, horizon loop.
  - **G1 for both:** each head must beat `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1) via the existing `floor_harness` on CRPS/coverage/Winkler, else the floor serves — the comparison harness IS the deliverable.
- [ ] **T0.5** Temper gate substrate (gate half only) — per-field residual pools + confidence readout (Bench-817 dispatch precedent) + fused-abstain composition (confidence ∧ distance) + null-with-guess emission. Sigmoid only.
- [ ] **T0.6** benches + gates — G2 perf bar (N-field request p99, CPU posture, pinned number — the scoreboard's PROJECTED rows #6/#7 and row #2's SPLIT become MEASURED here or are re-scored), G4 alloc gate (canary-armed), G1 floor comparison (T0.4's harness), **G5: external BLAKE3-pinned extraction fixture (cua-s1-arena pinning shape — never fixtures authored for the test) + TWO baselines: a Duckling/Recognizers-class rule floor AND a GLiNER-class trained lane; `span_pointer` earns GOAT only by beating the rule floor.** `required-features` rows for every gated target in the SAME commit (the repo-birth gate discipline).

## Phase 1 — Assembly (OWNER-GATED at Proposal 016 §7 — do not start without GO)

- [-] **T1.x** `riir-fathom` repo birth, F0 assembly, serving edge, harness suites, arena vs LLM-JSON lane — **deferred pending owner GO** (repo birth + name veto window).

## GOAT gate rule

Every feature promotes to default only via its bench + G1–G4 pass (G2 at `--release`). The estimate head's promotion criterion IS the floor comparison: beating the conformal-naive floor is the gate; losing to it demotes the head to opt-in with the floor served. UQ floor extension (Research 322 / Plan 340) applies unchanged. The span primitive's GOAT requires the G5 rule-floor win (verdict round 1: our own goldens alone are circular).

**Tier note (verdict round 1):** the family synthesis is GOAT (provisional), not Super-GOAT — re-score after T0.4 + T0.6 + the external G5 fixture report (Research 627 §8).
