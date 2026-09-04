# Bench 675 — `rating` core promotion (Issue 686): Elo/PL primitives, 4 copies → 1

**Date:** 2026-08-25 · **Repo:** katgpt-rs · **Issue:** `#686` (closed by this bench)
**Consumers:** katgpt-pruners arena `EloCalculator` · katgpt-pruners proof `lambda_to_elo` · riir-ai `riir-games` ruliology `ParadigmRanking` · riir-clippy `src/elo.rs` (Issue 039)

## Verdict

**PROMOTED to katgpt-core DEFAULT-ON** (`rating = []`, default list Phase 27).
Pure modelless arithmetic; the promotion axis is **DRY (4 → 1 implementations)**
with **bit-identical delegation** as the correctness gate — every consumer's
expression tree is preserved verbatim in `katgpt_core::rating`, so no test
expectation and no persisted rating changes anywhere.

Substrate-first found a FOURTH copy beyond the issue's three (riir-games
ruliology `ParadigmRanking` — win/loss/**draw**, the house tournament score
convention), which set the API shape: `update_scored` (score ∈ [0,1]) is the
canonical form; the binary `update` is its extremes (bit-identical, pinned).

## The primitive

`katgpt-core/src/rating.rs` (feature `rating`, default-on):

| Item | Notes |
|---|---|
| `STANDARD_K` / `STANDARD_BASE` / `STANDARD_SCALE` | 32 / 1200 / 400 — the standard-chess values every copy used (base-1200 = the PL `elo_offset`; arena + ruliology seed at 1000, a per-consumer choice) |
| `expected(a, b, scale)` | `1/(1+10^((b-a)/scale))` — verbatim union expression tree |
| `update(a, b, a_won, k, scale)` | binary win/loss (arena + riir-clippy form) |
| `update_scored(a, b, score_a, k, scale)` | `[0,1]` scored — adds the draw case (riir-games form) |
| `elo_from_lambda(λ, base, scale)` | the PL→Elo curve `base + scale·log10(max(λ,1e-10))` (clamp verbatim from `lambda_to_elo`) |
| `expected_f32` / `update_f32` | f32 twins — riir-clippy's persisted f32 ratings keep exact former numerics (no f64→f32 double rounding) |

The curve IS the bridge between the two rating systems: pairwise Elo's
equilibrium (expected == win rate) sits exactly on `elo_from_lambda`'s
`scale·log10(W/L)` — pinned by test in the module.

## GOAT

- **G1 (behavior preservation / bit-identity): PASS**
  - Core fixtures (9 tests): exact ±16 at base (both precisions, both bases),
    conservation <1e-9 (win/loss + draw), upset property, draw no-op at equal +
    the 400-gap drift values, λ fixtures (λ=1→base exactly, 10→+scale,
    0.1→−scale, clamp finite), binary==scored-extremes bit-identity,
    f32↔f64 agreement <0.01 over a 12-match sequence, equilibrium-on-the-curve.
  - katgpt-pruners: **126 default / 366 fft / 389 proof_sketch_evolution** —
    count-identical pre/post, all green (the 4 λ-conversion tests switched to
    the core call, expectations unchanged).
  - riir-games ruliology::ranking: **7/7** green (baseline held).
  - riir-clippy: **675 / 753 / 707** (default/latent_retrieval/ruliology_search)
    green, counts unchanged — the persisted f32 store ratings do not drift.
  - `bomber_tjs_arena` (EloCalculator literal consumer): compiles clean.
- **G2 (perf): PASS (cold path, intrinsic cost)** — release, 1M calls:
  `expected` 2.44 ns · `update` 1.70 ns · `update_scored` 0.79 ns ·
  `elo_from_lambda` 3.00 ns · `expected_f32` 1.77 ns. Delegation adds a
  call at most (LLVM inlines; update measured cheaper than expected via loop
  CSE). Rating math was never hot-path in any consumer.
- **G3 (no-regression): PASS** — katgpt-core default **1913 passed + 7
  ignored** (rating's 9 included; count reconciles 1920 total);
  `--no-default-features` and `--no-default-features --features rating`
  compile clean; **wasm32-unknown-unknown clean** (boundary contract audit
  target); clippy 0 on katgpt-core + katgpt-pruners (the one failure during
  the sweep was the PRE-EXISTING katgpt-percepta `len_zero` divergence —
  documented at T3-era, untouched by this change); riir-clippy clippy 0 in
  all three states (3 `cast_possible_truncation` allows on the f64→f32 const
  casts — 32/400/1200 are exactly representable, lossless by construction).
- **G4 (alloc-free): PASS by construction** — pure arithmetic, no allocation.
- **Modelless: PASS** — pure rating arithmetic, no deps, no training; the
  domain test from the issue holds (win/loss/draw conventions already the
  house standard at `induced_cwm/tournament.rs`).

## Consumer switches

| Consumer | Change |
|---|---|
| katgpt-pruners arena | `EloCalculator::expected/update` → `katgpt_core::rating::{expected,update}` (struct + `k`/`base` fields stay — public API; base seeds ratings, never enters the math) |
| katgpt-pruners proof | local `lambda_to_elo` DELETED; `rate_with_scratch` calls `elo_from_lambda`; the 4 conversion tests re-pointed (expectations unchanged) |
| riir-games ruliology | `ParadigmRanking::from_results` inline formula → `update_scored`; `ELO_K` = core `STANDARD_K` (seed 1000 stays local) |
| riir-clippy | `src/elo.rs` `expected_score`/`update_pair`/consts → core f32 twins + `STANDARD_*` casts; `EloTable`/`default_elo`/store surface unchanged (039's "switches to the core primitive when 686 lands" — done) |

## Non-goals held

- No online/incremental ELO API (riir-clippy 039 stays batch-on-load).
- No selection-semantics change (039 T5's ReportOnly verdict untouched).
- The Gibbs sampler stays in katgpt-pruners (single consumer; RNG-bound,
  not no_std-cheap) — it moves if/when a second PL rater consumer appears.

## Records

- katgpt-rs: `crates/katgpt-core/src/rating.rs` + lib.rs + Cargo.toml
  (feature + default Phase 27) + pruners delegation commits.
- riir-ai: riir-games ruliology delegation commit.
- riir-clippy: elo.rs delegation commit (+ AGENTS.md 039 switch note).
