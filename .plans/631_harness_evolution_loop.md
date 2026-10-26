# Plan 631: Harness Evolution Loop — Comparator Extraction (from the scenelab arena)

**Status:** Active — Phase 1 not started
**Date:** 2026-10-26
**Research:** [katgpt-rs/.research/637_Harness_Evolution_Loop_Primitive.md](../.research/637_Harness_Evolution_Loop_Primitive.md)
**Source paper:** [arXiv:2604.21003](https://arxiv.org/abs/2604.21003) — "The Last Harness You'll Ever Build" (two-level harness/meta-evolution framework)
**Rescope note (reviewer round 2):** this plan was rewritten from green-field primitives to an EXTRACTION. The scenelab arena (`riir-game-sdk/crates/riir-scenelab/src/arena.rs`, Proposal 029 Phase 4) already ships the comparator and the mutation-grammar pattern — this plan moves the rule to katgpt-core so the arena and every other consumer share ONE definition. The LCB does not move: `wilson_score_ci` + `beta_lcb*` already ship at `katgpt_core::hint_regret` and are consumed as-is.
**Adoption dependency:** riir-game-sdk Plan 009 T3.1 (the arena consumes the extracted comparator); riir-ai Plan 642 (the runtime consumer).
**Target:** `crates/katgpt-core/src/harness_evolution.rs` (new module) + Cargo feature `harness_evolution`

---

## Goal

Extract the arena's two-tier pairwise comparator — **exactly as it ships** (verdict-class rank Pass > Partial > Fail/Timeout → fired fraction → draw on identical; NO third tier) — into katgpt-core as `VerdictRank`/`two_tier_score`, so the arena's Elo tables and champion choice are byte-identical before and after adoption. The mean-terminal-tick axis becomes an OPT-IN third tier (`two_tier_score_with_ticks`) that ONLY the runtime consumer (riir-ai Plan 642) uses — adopting it in the arena would flip current draws to wins/losses and change champion selection, which sdk Plan 009's byte-identical promise forbids. Also expose the mutation-grammar seam (`MutationArms` — CONDITIONAL, same rule as `BestKeeper`) and land `BestKeeper<H>` ONLY with committed consumers. GOAT gate: zero-alloc microbench + total-order-stable comparator; **promotion is adoption-gated** — the primitive promotes only when the arena (sdk Plan 009 T3.1) and Plan 642 actually consume it.

## Phase 1 — Extraction (CORE)

### Tasks

- [ ] **T1.1** `VerdictRank`/`two_tier_score` — the arena's pairwise comparator EXACTLY (verdict class → fired fraction → draw; first-index tie rule per the determinism law); the arena's inline rule is the SPEC and adoption must leave Elo tables + champion byte-identical; unit tests for boundary/tie/draw cases. `two_tier_score_with_ticks` — the opt-in third tier (mean terminal tick, budget-normalized) for the runtime consumer (Plan 642) only; the arena never consumes it.
- [ ] **T1.2** `BestKeeper<H>` — CONDITIONAL: land only with ≥2 committed consumers (arena generations via sdk Plan 009 + riir-ai Plan 642; optionally riir-refine `EvolveRecorder`); bounded fixed-capacity history ring, zero-alloc iteration, evolve-from-best read, full-history digest for chaining. If consumers do not commit, this stays consumer-local and is dropped from the plan.
- [ ] **T1.3** `MutationArms` trait — CONDITIONAL (same rule as T1.2): the arena's ladder+clamp+seed pattern generic over dial domains — bounded per-arm grammar, seed-pinned candidate stream, deterministic lowest-index tie-break. Land in katgpt-core only once ≥2 consumers commit (arena + Plan 642); with one consumer (the arena) it stays in scenelab. Selection glue REUSES `hint_regret::beta_lcb_order_into` + `wilson_score_ci` — no new LCB in this module.
- [ ] **T1.4** Feature `harness_evolution` (opt-in); no default-feature surface change; doc example wiring a toy worker/evaluator.

## Phase 2 — GOAT Gate

### Tasks

- [ ] **T2.1** Zero-alloc microbench (`benches/harness_evolution_goat.rs` behind the feature): comparator + keeper path allocation count = 0; total-order stability across 10⁶ shuffles of tied inputs.
- [ ] **T2.2** Verdict note in `.benchmarks/` (enabling primitive — consumer gain benches referenced, never restated). **Promotion rule: adoption-gated** — promote to the arena/642 consumers' default surface only after their benches land; otherwise stays opt-in.
