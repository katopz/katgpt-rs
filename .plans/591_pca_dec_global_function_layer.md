# Plan 591: PCA Global-Function Layer — DEC Aggregates Wired into the CA Decision Function

**Status:** OPEN — Phase 0 not started. Open primitive from Research 544 ([arXiv:2609.06102] distillation). Game application is riir-ai's (Proposal 012 revival, Issue 911) — this plan ships ONLY the generic katgpt-dec composition layer, no game semantics.

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

- [ ] `pca.rs` module in `katgpt-dec` behind `pca_global = ["dep:..."]` (no new external deps; re-export via `katgpt_core::dec`)
- [ ] `PcaGlobalFn` enum: `Betti0` / `BoundaryFluxMass` / `BeliefMassDivergence` / `Codifferential` — each evaluates against the state cochain + `CellComplex` via existing operators; `debug_assert!(dim() <= 3)` on every boundary-flux path
- [ ] `PcaDecision` trait: `decide(&local_neighborhood: &[f32], &globals: &GlobalScalars) -> f32` — decision consumes ONLY kernel outputs + global scalars (paper's constraint: decision never reads raw state)
- [ ] Seed implementation: birth-death alive gate + global termination term ("stop placing when count ≥ k" — the construct pure-local CA cannot express)
- [ ] `step_pca_sync(...)`: compute globals once per iteration (via `DecCache`), then sweep cells; `*_into` scratch variants, zero-alloc per tick
- [ ] Unit tests: decision-input purity; d-guard fires; flag-off builds to nothing (required-features row + `#![cfg]` per the green-zero rule)

## Phase 1 — Async global feedback (the paper's dynamics)

- [ ] `step_pca_async(...)`: FIXED row-major traversal; O(1) incremental counters updated on each cell write (swarm `ripe_per_band` pattern) — deterministic by traversal order
- [ ] Sync vs async comparison test: async must avoid the stale-count over-placement failure (paper §3 counter example) on a k-target placement task
- [ ] G1 determinism assertion: same seed + traversal → bit-identical final grid (property test, ≥100 seeds)

## Phase 2 — Largest-component size (the one crosswalk gap)

- [ ] Extend globals with `LargestComponentSize`: b0 gives component *count* only; try harmonic-projector support extraction first, fall back to a committed single scan; keep closed-form if achievable, else document the O(n) scan honestly
- [ ] Crosswalk parity test: on synthetic grids, `BoundaryFluxMass` == perimeter count, `Betti0` == component count, `Codifferential` sign == clustering direction (the Research 544 §2.1 table as executable spec)

## Phase 3 — GOAT gate + bench

- [ ] Bench file `.benchmarks/` per numbering: three competitors on a 64×64 tile map from seed — (a) pure-local `birth_death`, (b) sync `step_pca`, (c) async `step_pca` — plus static-generator incumbent
- [ ] **G1 correctness:** final-state connectivity verified by an *independent* `betti_numbers` call; determinism per Phase 1
- [ ] **G2 perf:** iterations-to-b0==1 ≥3× better than pure-local (paper suggests 3–8×; Sokoban 98→19)
- [ ] **G3 no-regression:** flag-off path bit-identical to current `birth_death` output on shared test grids
- [ ] **G4 alloc:** zero allocation per tick (scratch-only)
- [ ] GPU-exclusivity noted in the bench doc if any gate run overlaps compute work (AGENTS.md rule); Unity/Zed exempt
- [ ] GOAT pass → promote `pca_global` to default; fail → stay opt-in with raw numbers recorded in the bench doc
