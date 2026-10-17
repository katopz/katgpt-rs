# Bench 944 — katgpt-assign Phase 2: escape stages + Phase 2 spec vocabulary

> **Status:** RECORD — Phase 2 quality evidence; wall-time column NOT A MEASUREMENT (see provenance)
> **Plan:** 620 (katgpt-assign Phase 2 — specs/moves/collapse/explain/policy)
> **Date:** 2026-10-17
> **Run:** `cargo bench -p katgpt-assign --bench bench_924_assignment_g2_baselines` (release)

## Provenance (the latency-claim rule)

`../riir-reflex/scripts/bench_preflight.sh` **REFUSED** this run:
`power=AC Power load=6.18 swap=7318.62M powermode=2(high)` — load over MAX_LOAD=6.0
(sibling sessions on the box) and High Power Mode. Per the rule, **no wall_ms
number below is publishable**; they are shown only for relative order-of-magnitude
context. The quality columns (folded objective, violation root, spread, moved,
max_util, evals) are deterministic solver outputs — box-independent and exact
(G1: integer determinism, oracle-pinned by tests).

## The headline: escape stages break the partition-closure stall

The perfect_balance family has seeds whose arbitrary-part packing is unreachable
by ANY single/swap sequence (the fixture doc's recorded "partition closure can
require triple moves"). Measured on seeds {4, 10, 27, 34}: plain local search
stalls at violation root 100 (= FIX_IT_WEIGHT 100 × 1 unit of capacity excess);
both Phase 2 escape stages — TRIPLE_LOOP rotations and KL_SEARCH chains — reach
violation **0** on every stalled seed:

| seed | plain violation | +triple | +kl |
|---|---|---|---|
| 4 | 100 | **0** | **0** |
| 10 | 100 | **0** | **0** |
| 27 | 100 | **0** | **0** |
| 34 | 100 | **0** | **0** |

Escape-stage cost is marginal: evals rise ≤ 4% over the plain run (the escape
runs only at the stall — the run loop's escape-stage placement), and the stage
stops as soon as the stall is broken. Wall_ms rows under the refused preflight:
plain 3.9–38.9 ms, +3cyc 4.1–38.96 ms, +kl 4.62–40.26 ms — the escape stage
roughly doubles the STALL-LEG cost only because the plain leg wastes its whole
budget on an unreachable repair; treat as context, not measurement.

## Phase 2 spec vocabulary at mid size (500 objects × 20 containers)

Full Phase 2 spec set (Capacity + GroupCount over 100 members + AvoidMoving
pins + 50 AssignmentAffinities + MinimizeMovement): the default posture reaches
**feasibility (violation 0)**; the all-stages posture (triple 20k + KL 4 +
single_sample 8) reaches the same folded objective (0) — the escape stages add
nothing once the default posture is feasible, exactly as designed (they are
stall escapes, not quality boosters).

## Never-worse property (the escape-stage contract)

`escape_stages_never_end_worse_than_plain` (tests/phase2_moves.rs): the armed
run shares the plain run's trajectory until the first stall (deterministic
search) and every escape accept is δ ≤ 0 from there — final folded objective
≤ plain's, per instance, deterministically. Pinned over seeds 1..=4 on the
random family + the directed stall family above.

## Phase 1 rows unchanged

The Phase 1 table re-ran green in the same invocation (initial/ffd/local rows
identical in kind to `.benchmarks/924_assignment_g2_baselines.md`: local
dominates FFD 3–47× on the random family, exact optima on the known-optimum
families). The Phase 2 changes are additive: new specs compile to new leaves;
the default Limits posture is byte-identical to Phase 1 on Phase 1 spec sets
(single_sample/triple_budget/max_kl_passes default OFF; Policy::Default = the
Phase 1 measured posture) — pinned by the full Phase 1 suite (9 determinism +
5 correctness + 5 optimality + 1 lower-bound + G4 alloc both postures).

## Verdict

Phase 2 GOAT posture: **feature stays OPT-IN (`assignment`), quality gains
proven, no default-path change** — the promotion criterion (a consumer
promoting `shard_solver` to its own default) is unchanged from Plan 620's
verdict protocol.
