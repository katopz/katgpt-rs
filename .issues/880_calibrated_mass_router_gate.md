# Issue 880 — calibrated-mass router gate: the `exact_mass_admit` consumer lane for `gate_sigmoid_topk` (Bench 884 promotion candidate (a))

**Status:** OPEN — filed 2026-09-24 (the 4090 session, from Bench 884's recorded
promotion candidates). Design-first; the IMPLEMENTATION + bench serialize
against the concurrent 4090 windows (the reflex Issue-018 harness run occupies
the box until it exits — no katgpt-rs release builds until then).

## Source

[Bench 884](../.benchmarks/884_exact_mass_admit_goat.md) §Verdict records four
consumer postures for the calibrated-mass family; this issue takes the cheapest
one: **the calibrated-mass upgrade of `gate_sigmoid_topk`** (katgpt-spectral
router gates). Finding 2 of that bench: ours is already CHEAPER than the
shipped uncalibrated gate in its designed regime (69.8 µs vs 77.1 µs at
N=1e3, 0.906×) — the upgrade is not a perf trade, it is accuracy-for-free at
game-scale N.

## What is being added (and what is NOT)

A **new, separately-named gate function** in
`katgpt-spectral/src/manifold_power_iter_router.rs` — e.g.
`gate_sigmoid_topk_mass[_into]` — that keeps the exact-k selection-sort
indices (the contract consumers rely on: exactly `kk = min(k, n)` indices in
descending score order) but emits **calibrated masses**: `mᵢ =
σ((sᵢ − τ)/T)` with `τ` solved by `katgpt_core::exact_mass_admit_into` so
that `Σᵢ∈all mᵢ = k` (the tail's sub-0.5 mass is the leakage term; with
well-separated scores it is negligible, with degenerate scores the gate
honestly refuses to hand out full mass).

**NOT a drop-in replacement — semantic difference is the point (and the
trap):**

- `gate_sigmoid_topk`'s scores are per-expert INDEPENDENT (t06 pins this:
  perturbing one expert's router row cannot move another's score — sigmoid,
  not softmax).
- The calibrated mask couples every expert through `τ` (perturbing one row
  moves `τ`, which moves every mass). t06's byte-identity assertion FAILS for
  the new function BY CONSTRUCTION — the tests must therefore be
  suite-separated, never shared. The `exact_mass_admit` module doc already
  pins the naming split ("every consumer of exact mass says
  `exact_mass_admit`, every consumer of hard cut says `gate_sigmoid_topk`").

## Gates (the feature-flag discipline — GOAT before any default talk)

- `G1 correctness`: selection-equivalence (same index set + order as
  `gate_sigmoid_topk_into` on seeded fixtures — τ only reweights, never
  re-ranks: σ is monotone in s) + mass calibration (`|Σm − k| ≤ 1.2e-7·n`
  class, the Bench-884 G1 bound) + the t06-independence expectation INVERTED
  for the new function (perturbing one row MAY move other masses — pinned as
  a documented property, not left as a surprise).
- `G2 perf`: the new lane must not be slower than the incumbent at its
  designed N ≤ 1e3 regime (Bench 884 says 0.906× is available; the gate is
  ≤ 1.05× to allow wiring noise, else investigate).
- `G3 no-regression`: every existing router test (t06, t10, t11, the GOAT
  benches) passes byte-identical — the new function is additive; nothing
  existing changes behavior.
- `G4 alloc`: `_into` form zero-alloc (caller-owned scores + idx buffers; τ
  bisect is scalar work).

Feature posture: the function lands behind katgpt-spectral's existing
`manifold_power_iter_router` feature + a katgpt-core dependency edge on
`exact_mass_admit` (already shipped, opt-in). Promotion to any default
surface requires a CONSUMER that measurably gains (the router today has no
production consumer outside the crate's own benches — the promotion question
is deferred until one exists; this issue only adds the gated primitive).

## Consumers to check before landing

`gate_sigmoid_topk[_into]` callers today: the router module itself, its bench
(`manifold_power_iter_router_bench`), and `quantile_balance_router.rs`'s
pattern comment (it runs its own selection sort — not a caller). No riir-*
consumer exists (workspace grep 2026-09-24). The new function is additive —
no caller migrates in this issue.

## Box-state constraint

No release builds on this box while the reflex Issue-018 harness run lives
(the serialization discipline — one CPU-heavy job; the run started 11:16:35
+07 and is expected to run for hours). The implementation commit can be
authored; `cargo bench`/`full_gate` verification waits for the window.
