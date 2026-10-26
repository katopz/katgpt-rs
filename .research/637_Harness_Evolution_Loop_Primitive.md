# Research 637: Harness Evolution Loop — Generic Primitive Stub

> **Source:** [The Last Harness You'll Ever Build](https://arxiv.org/abs/2604.21003) — Seong, Yin, Zhang, Shi (Sylph.AI), arXiv:2604.21003, v3 2026-05-01
> **Date:** 2026-10-26
> **Status:** Active — stub; primary distillation lives in the private tier
> **Related Research:** 145 (SIA harness/weight co-evolution stub — same family)
> **Related Plans:** [Plan 631 — harness evolution loop primitive](../.plans/631_harness_evolution_loop.md)
> **Cross-ref (riir-ai):** [Research 401 — full guide + fusion verdict](../../riir-ai/.research/401_Harness_Evolution_Game_Autoplay_Guide.md) (SUPER-GOAT verdict, prior-art landscape, Path 0 inventory, consumer reframes)
> **Classification:** Public

---

## TL;DR

arXiv:2604.21003 automates harness engineering with two loops: a Harness Evolution Loop (Worker → adversarial Evaluator → Evolution agent, monotonic best-keeping over full history) and a Meta-Evolution Loop (evolves the blueprint Λ = (W, H⁽⁰⁾, V, E) itself across tasks — meta-learning over the evolution process). No training anywhere; the only mutation operator is an LLM editing prompts/code.

**Distilled for katgpt-rs (modelless, inference-time):** the loop MATH is generic and modelless-shaped — a bounded candidate vector, a deterministic two-tier scorer (pass/fail then time), a monotonic best-keeper over a bounded history ring, and an outcome-driven arm selector over a mutation grammar. The paper's LLM slots (evaluator, evolver) are consumer-supplied; the substrate stays zero-LLM, zero-grad.

**This repo ships the generic primitive; the game-runtime fusion (harness-as-latent-vector for autoplay + self-evolve, guards, meta level) is the private tier's verdict — read Research 401 before implementing anything here.** The tier call (SUPER-GOAT at the modelless game-runtime mechanism level, with the honest class-level prior-art caveat: DGM arXiv:2505.22954, HarnessCompass arXiv:2608.01918, JIT-Agent arXiv:2608.25593, ScholarEvolve arXiv:2609.40169 populate the LLM-driven class) lives in the cross-referenced note.

## The Open Primitive (Plan 631 scope — EXTRACTION from the shipped scenelab arena, not green-field)

At reviewer round 2 the shipped substrate was found: `riir-game-sdk/crates/riir-scenelab/src/arena.rs` (Proposal 029 Phase 4) already ships one full Level-1 round — seeded clamped variant mutations (`StrategyVariantSet::from_corpus`, `seed ^ variant_index` LCG), UCB1/round-robin tournaments over `ScenarioHarness`, pairwise Elo via `katgpt_core::rating` (verdict class then fired fraction — the paper's two-tier score in pairwise form), champion `DirVecPatch` PROPOSAL (apply path deliberately absent), `ArenaFloors` regression gate. This repo's share is therefore an EXTRACTION:

- `VerdictRank`/`two_tier_score` — the arena's pairwise comparator EXACTLY (verdict class → fired fraction → draw; no third tier) moved to katgpt-core as ONE definition; the arena consumes it (riir-game-sdk Plan 009 T3.1) with byte-identical Elo tables + champion. `two_tier_score_with_ticks` (mean terminal tick) is an OPT-IN third tier only the runtime consumer (riir-ai Plan 642) uses.
- `MutationArms` — the arena's ladder+clamp+seed pattern generic over dial domains; CONDITIONAL on ≥2 consumers (else stays in scenelab); selection glue REUSES `hint_regret::wilson_score_ci`/`beta_lcb_order_into` (already shipped and tested here) — no new LCB.
- `BestKeeper<H>` — CONDITIONAL on ≥2 committed consumers (arena generations + riir-ai Plan 642), else consumer-local.
- Feature `harness_evolution` (opt-in); GOAT gate = zero-alloc + total-order-stable comparator; **promotion is adoption-gated**.

The multi-generation chain, custody, and meta level are the consumers' work (riir-game-sdk Plan 009 + riir-ai Plan 642) — the GOAT verdict and full fusion live in the cross-referenced private note.

## What this repo does NOT take

LLM-in-the-loop evolution and free-form harness rewriting (the private tier's guard laws forbid both); the meta level stays a consumer composition (ESC grammar + never-lever guards live in the private tier's surfaces).
