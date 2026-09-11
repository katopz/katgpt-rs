# Plan 591: PCA Global-Function Layer — DEC Aggregates Wired into the CA Decision Function

**Status:** CLOSED — Phases 0–3 ALL LANDED (2026-09-11, this repo). Phase 2: `LargestComponentSize` variant + crosswalk parity (6 new tests, 249 armed). Phase 3: Bench 708 G1–G4 ALL PASS → `pca_global` PROMOTED TO DEFAULT in katgpt-dec (G2 iteration collapse 12.20× ≥ 3× gate; katgpt-core passthrough stays opt-in, se2 precedent). Gate row `katgpt-dec:249:pca_global`. Open primitive from Research 544 ([arXiv:2609.06102] distillation). Game application is riir-ai's (Proposal 012 revival, Issue 911) — this plan ships ONLY the generic katgpt-dec composition layer, no game semantics.

**Date:** 2026-09-10
**Source research:** [katgpt-rs/.research/544_Programmable_CA_DEC_Global_Function_Layer.md](../.research/544_Programmable_CA_DEC_Global_Function_Layer.md)
**Repo:** katgpt-rs (public) — `crates/katgpt-dec`
**Feature flag:** `pca_global` (opt-in; promote to default only on GOAT pass)

---

## Why

Programmable Cellular Automata (arXiv:2609.06102) measured that one global function
(count / connectivity) collapses CA iteration counts 3–8× and unlocks domains
purely-local rules cannot solve at all (Zelda 0% → 99% playability). The stack
already ships the discovered global set closed-form (`betti_numbers`,
`boundary_flux_mass`, `belief_mass_divergence`, `codifferential`) and the local
kernel (`stochastic_birth_death_step`, Plan 454) — but nothing wires a global
aggregate into a per-cell decision function. That composition is this plan.
Constraint: boundary-vs-volume evaluation is a win **only for d ≤ 3** — every
boundary-flux path asserts this.

## Phase 0 — Types + sync globals

- [x] `pca.rs` module in `katgpt-dec` behind `pca_global = ["grid_3d"]` (no new external deps; re-export via `katgpt_core::dec`)
- [x] `PcaGlobalFn` enum: `Betti0` / `BoundaryFluxMass` / `BeliefMassDivergence` / `Codifferential` — each evaluates against the state cochain + `CellComplex` via existing operators; `debug_assert!(dim() <= 3)` on every boundary-flux path *(Betti0 is the SUPPORT-aware union-find — `betti_numbers(cx)` is state-blind; pinned vs a BFS reference on random fields)*
- [x] `PcaDecision` trait: `decide(&local_neighborhood: &[f32], &globals: &GlobalScalars) -> f32` — decision consumes ONLY kernel outputs + global scalars (paper's constraint: decision never reads raw state) *(enforced by the signature: no parameter reaches the rest of the field; purity pinned by a recording-decision test)*
- [x] Seed implementation: birth-death alive gate + global termination term ("stop placing when count ≥ k" — the construct pure-local CA cannot express) *(GlobalTargetGate Above/Below; a tripped gate refuses every birth, existing cells keep their local dynamics — pinned e2e)*
- [x] `step_pca_sync(...)`: compute globals once per iteration (via `DecCache`), then sweep cells; `*_into` scratch variants, zero-alloc per tick *(DecCache deliberately deferred — no Phase-0 global needs a Hodge decomposition; all paths use the `_into`/`_scratched` operator variants through `PcaScratch`)*
- [x] Unit tests: decision-input purity; d-guard fires; flag-off builds to nothing (required-features row + `#![cfg]` per the green-zero rule) *(14 tests; the module is `#[cfg(feature)]` in lib.rs so flag-off compiles it to nothing — measured 225 default vs 239 armed; no `[[test]]` target so no required-features row applies; the gate ROW `katgpt-dec:239:pca_global` executes the armed surface)*

### Phase 0 en-route findings (2026-09-10)

1. **The d∘d=0 degeneracy (design-changing, pinned as a regression test):** the
   `BoundaryFluxMass` arm MUST NOT use the gradient lift `d(morph)` — the flux of
   a gradient around ANY closed boundary is identically zero by `d∘d = 0`. The arm
   ships the ENDPOINT-SUM edge lift `f[e] = m(tail)+m(head)` (the canonical
   pushforward under identity Hodge stars; not a gradient — its curl survives),
   measuring rim-vs-interior morphogen. Stokes identity hand-checked at 40.0 on
   the x-ramp probe + pinned executable (`flux_matches_the_volume_integral_stokes_identity`).
2. **`betti_numbers(cx)` cannot serve as the CA's Betti0** — it ranks the FULL
   complex's boundary matrices (a solid grid gives β₀=1 regardless of which cells
   are alive). The support-aware count has no closed form without restricted
   Gaussian elimination (far worse than O(V+E)), so it ships as union-find over
   the B₁ incidence, matched against a naive BFS reference on 25 random fields.

## Phase 1 — Async global feedback (the paper's dynamics)

- [x] `step_pca_async(...)`: FIXED row-major traversal; O(1) incremental counters updated on each cell write (swarm `ripe_per_band` pattern) — deterministic by traversal order *(incremental scope is HONEST: only `AliveCount` — added to the enum as the paper's count global — is O(1)-incrementable on births AND deaths; other arms degenerate to sync semantics, documented + pinned by an async(Betti0)==sync(Betti0) test; `Betti0` birth-incremental + death-dirty-resync hybrid deferred until a consumer needs it)*
- [x] Sync vs async comparison test: async must avoid the stale-count over-placement failure (paper §3 counter example) on a k-target placement task *(k=5: sync overshoots (batch gated against frozen count 2), async lands EXACTLY 5; live counter cross-checked against a fresh full evaluation)*
- [x] G1 determinism assertion: same seed + traversal → bit-identical final grid (property test, ≥100 seeds) *(100 seeds × 3 ticks)*

## Phase 2 — Largest-component size (the one crosswalk gap)

- [x] Extend globals with `LargestComponentSize`: b0 gives component *count* only; try harmonic-projector support extraction first, fall back to a committed single scan; keep closed-form if achievable, else document the O(n) scan honestly *(DESIGN CALL — coordinator: shipped as the EXISTING Betti0 support union-find extended with per-root component-size tracking, ONE O(V+E) pass, zero new deps — NOT a harmonic projector: per-component Hodge solves are strictly worse than O(V+E), so the union-find extension IS the "closed-form-if-achievable" answer; deviation documented here and in Bench 708. Pinned: basics (empty→0, single run→3, max-wins, bridge→6), the larger-component-wins case, and the same 25-random-field BFS harness as the Phase 0 Betti0 pin, now for both axes)*
- [x] Crosswalk parity test: on synthetic grids, `BoundaryFluxMass` == perimeter count, `Betti0` == component count, `Codifferential` sign == clustering direction (the Research 544 §2.1 table as executable spec) *(three pinned tests; honest verdicts — `BoundaryFluxMass` is the SIGNED DOMAIN-RIM measure `Σ_v m(v)·w(v)` (= 2× the covered-rim edge count; the plan's "perimeter count" needed exactly this factor 2, pinned not adjusted; an INTERIOR blob contributes ZERO — the paper's blob-perimeter is a different quantity, closest shipped signals are the divergence arms). `Codifferential` is a NORM and no signed divergence-direction global can exist (`Σ_v δf ≡ 0` by conservation, pinned): pinned instead bit-equality on complementary peak/dip profiles + strict norm decrease under heat smoothing (clustering = HIGH norm, dispersed = LOW). Full derivations in Bench 708)*

## Phase 3 — GOAT gate + bench

- [x] Bench file `.benchmarks/` per numbering: three competitors on a 64×64 tile map from seed — (a) pure-local `birth_death`, (b) sync `step_pca`, (c) async `step_pca` — plus static-generator incumbent *(Bench binary `benches/bench_591_pca_global_goat.rs` (plan-numbered per repo convention); DOC = `.benchmarks/708_pca_global_goat.md` — 707 was taken by the concurrent saddle-trap session at write time; highwater bumped 707→708. Static incumbent = one fixed 33-cell connected line, 0 iterations, context row)*
- [x] **G1 correctness:** final-state connectivity verified by an *independent* `betti_numbers` call; determinism per Phase 1 *(SUPERSEDED per Phase 0 finding — `betti_numbers(cx)` is state-blind; the independent reference is the BFS traversal pinned in Phase 0's tests, mirrored as `bfs_components` in the bench. Cited in Bench 708. Determinism: 100 seeds × 3 arms bit-identical — PASS)*
- [x] **G2 perf:** iterations-to-b0==1 ≥3× better than pure-local (paper suggests 3–8×; Sokoban 98→19) *(PASS at **12.20×** — 61 pure-local fixpoint ticks vs 5 pca halt ticks, median of 10 seeds. GATE-READING DEVIATION documented in Bench 708: the literal first-tick-to-b0==1 comparison is 1.00× BY CONSTRUCTION (identical pre-trip dynamics, trip costs +1 tick); the gate is evaluated at each arm's termination point — the early stop IS the global function's value — with the raw 1.00× diagnostic printed every run. No bar adjustment)*
- [x] **G3 no-regression:** flag-off path bit-identical to current `birth_death` output on shared test grids *(in-bench: 10 untripped `step_pca_sync` ticks bit-identical to 10 stock kernel ticks, same seed — PASS; external: flag-off config `--no-default-features --features heat_kernel_trajectory,sheaf_admm,grid_3d,se2_equivariant_lift` = exactly 225 lib tests, the pre-change default count — compile-to-nothing proof holds; `birth_death.rs` untouched by Phases 2–3)*
- [x] **G4 alloc:** zero allocation per tick (scratch-only) *(PASS: 0 allocs / 100 ticks for BOTH step paths, CountingAllocator)*
- [x] GPU-exclusivity noted in the bench doc if any gate run overlaps compute work (AGENTS.md rule); Unity/Zed exempt *(N/A noted in Bench 708 — CPU-only arena, no GPU compute overlap)*
- [x] GOAT pass → promote `pca_global` to default; fail → stay opt-in with raw numbers recorded in the bench doc *(PROMOTED 2026-09-11 — all gates pass: katgpt-dec default list + comment updated with the Bench 708 verdict; README default-on count 197→198 (totals re-measured at 587, includes the concurrent sibling promotion); catalog §98 → DEFAULT-ON; test gate row 243→249. The katgpt-core `pca_global` passthrough stays opt-in there (dec_operators layer split, se2_equivariant_lift precedent))*
