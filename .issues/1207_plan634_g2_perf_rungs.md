# Issue: Plan 634 G2 perf rungs — per-edge division hoist + (τ+1)-ball restriction

**Date:** 2026-11-12
**Status:** Open — filed by the T2.6 G2 measurement (bench `plan634_div_g2_latency`, this commit)
**Origin:** katgpt-rs Plan 634 T2.6 (the G2/G4 gate); the measured ladder refuted the module doc's "fixture-scale G2 latency is µs either way" premise.

## The measured facts (release, τ=4, amo under disclosed load — mins quoted)

- `diffusion_similarity_local` planted(8,8) n=64: **min 72 µs** (mean 84 µs) vs the plan bar ≤1 µs / ≤10 µs — the battery-scale FAIL.
- `div_tau` planted n=64: min 35 µs; circulant n=64: min 1.4 µs — **edge-density-driven** (E-linear), sparse ladders stay µs-class end to end (n=1000 circulant: 22 µs).
- `neighbor_diversity_pick`: **22.8 ms at k=23 (planted n=64), 1.39 s at k=47 (planted n=256)** — the k² kernel assembly inherits similarity's per-edge cost; the walk-time selector is per-step-unusable at planted degree.
- Spectral query paths: **40 ns / 17 ns** — the only strict-bar PASS. `DiffusionEmbedding::build` 1.3 ms (n=64) → 362–602 ms (n=256) — the Jacobi O(n³) dominates `batch_select_diffusion` (~0.5 s/call at n=256, per-batch class).

Root cause of the matvec cost: `diffuse_into` / `div_tau` compute `w / degree(z)` — an **f32 division per edge per matvec** (~6.8 ns/edge measured ≈ 20+ cycles) — and the walk pick assembles k² similarity calls per pick.

## Task

- [ ] **R1 (bit-exact hoist, small):** precompute the per-edge normalized weight `w/deg(z)` into a parallel CSR array at `SimGraph::build` (the ratio is a build-time constant per edge); both matvec loops multiply instead of divide. Bit-identical results (same operands, same f32 division, folded once; accumulation order unchanged — the Issue-940 pin holds). Expected ~3–5× on every matvec-class number. Re-run the ladder + the full plan634 battery (bit-exactness fixture: T2.2's local==spectral equality at 1e-6 must hold unchanged; F1's closed form 1.8e-9 likewise).
- [ ] **R2 (algorithmic, the paper's own efficiency half):** implement the (τ+1)-ball restriction for `neighbor_diversity_pick`'s kernel assembly — all its queries sit within 2 hops of `x`, so each similarity is O(|B|·deg) over a ~100-node subgraph instead of O(τ·E) over the full graph. This is the rung the module doc reserved ("lands only if a consumer's profile demands it") — the T2.6 measurement IS the demand: the walk pick must reach ms-class at planted degree before any walk-time consumer (riir-ai Plan 646 family) can arm it per-step.
- [ ] **R3 (record-only note):** `batch_select_diffusion`'s ~0.5 s/call at n=256 is admission-cycle-acceptable (Plan 646 wires it per cycle, not per tick) — no work item unless a consumer tightens the cycle budget; revisit only with a named consumer bar.
- [ ] **R4:** after R1/R2 land, re-run the ladder and re-pin the bench's battery-scale sanity ceilings (currently 10× the T2.6-measured values, derivation in the bench source) and re-adjudicate the plan's ≤1 µs / ≤10 µs bar in plan 634's status line.

## Gates

- R1 must be bit-exact: the T2.2 battery + T2.1 closed-form pin are the reproduction surfaces; any drift = loud FAIL, never a tolerance widening.
- G4 zero-alloc census stays green across both rungs (a parallel CSR array adds build-time memory only; R2's ball work buffers are caller-scratch or build-time).
- The feature stays opt-in throughout; no promotion claim rides either rung (the plan's G2 verdict is already recorded FAIL — these rungs are the re-open path, not a promotion).
