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

- [x] **R1 (bit-exact hoist, small):** LANDED 2026-11-13 — `SimGraph` carries a parallel build-time-normalized CSR `w_norm` (`row_normed()`, parallel to `row()`); the three matvec loops (`diffuse_transpose`, the local-similarity `diffuse_into`, `div_tau`'s inner fold) multiply instead of divide-and-re-walk. **The win EXCEEDED the ~3–5× estimate because the real hot-loop shape was O(E·d̄), not O(E): `degree(z)` re-walked row z's weights per edge visit — the hoist killed both the per-edge division AND the hidden quadratic degree walk.** Ladder (min ns, pre-R1 → post-R1): planted n=64 sim 72 209→3 958 (**18×**), div 35 459→1 709 (**21×**); circulant n=64 sim 2 833→875 (**strict ≤1 µs PASS**), div 1 375→417; planted n=256 sim 1 136 083→31 500 (**36×**); planted n=1000 sim 39 322 791→405 791 (**97×**); circulant n=1000 sim 44 916→13 375; walk pick k=23 21 658 542→1 083 917 (**20×**, now ms-class), k=47 1 338 712 083→38 843 625 (**34×**); spectral unchanged (reads its own arrays, expected). **Bit-exactness: the FIRST pass RED the T2.2 1e-6 pin (err 1.116e-6 on planted-K6-M5 τ=1) — the divisor used the build's edge-order `deg` accumulation while the historical call-time `degree()` folds row-CSR ascending-slot order; f32 non-associativity ⇒ ULP-divergent divisors. Fixed: the build now folds `deg_csr` exactly as `degree()` does (same order, same operands) — all 9 plan634 targets green (18/18) incl. the bit-pinned t14 (μ=π ⇒ exactly 1.0). The ULP lesson recorded here because it is the corpus's working example of "same value, different fold order" breaking a bit-exactness claim.**
- [ ] **R2 (algorithmic, the paper's own efficiency half):** implement the (τ+1)-ball restriction for `neighbor_diversity_pick`'s kernel assembly — all its queries sit within 2 hops of `x`, so each similarity is O(|B|·deg) over a ~100-node subgraph instead of O(τ·E) over the full graph. This is the rung the module doc reserved ("lands only if a consumer's profile demands it") — the T2.6 measurement IS the demand: the walk pick must reach ms-class at planted degree before any walk-time consumer (riir-ai Plan 646 family) can arm it per-step.
- [ ] **R3 (record-only note):** `batch_select_diffusion`'s ~0.5 s/call at n=256 is admission-cycle-acceptable (Plan 646 wires it per cycle, not per tick) — no work item unless a consumer tightens the cycle budget; revisit only with a named consumer bar.
- [ ] **R4:** after R1/R2 land, re-run the ladder and re-pin the bench's battery-scale sanity ceilings (currently 10× the T2.6-measured values, derivation in the bench source) and re-adjudicate the plan's ≤1 µs / ≤10 µs bar in plan 634's status line.

## Gates

- R1 must be bit-exact: the T2.2 battery + T2.1 closed-form pin are the reproduction surfaces; any drift = loud FAIL, never a tolerance widening.
- G4 zero-alloc census stays green across both rungs (a parallel CSR array adds build-time memory only; R2's ball work buffers are caller-scratch or build-time).
- The feature stays opt-in throughout; no promotion claim rides either rung (the plan's G2 verdict is already recorded FAIL — these rungs are the re-open path, not a promotion).
