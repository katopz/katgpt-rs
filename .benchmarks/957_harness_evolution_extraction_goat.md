# Bench 957 — `harness_evolution` Extraction GOAT Gate (Plan 631 T2.2)

**Status:** RECORD
**Date:** 2026-10-26
**Plan:** [631_harness_evolution_loop.md](../.plans/631_harness_evolution_loop.md) (comparator EXTRACTION from the scenelab arena)
**Research:** [637_Harness_Evolution_Loop_Primitive.md](../.research/637_Harness_Evolution_Loop_Primitive.md)
**Vehicle:** `cargo bench -p katgpt-core --features harness_evolution --bench harness_evolution_goat` (harness = false, exit-code verdict)

## What this gates

An **extraction**, not a new mechanism: `VerdictRank` / `two_tier_score` /
`two_tier_score_with_ticks` / `champion_index` are the riir-scenelab
scenario arena's shipped comparator, moved verbatim into katgpt-core so
the arena (sdk Plan 009 T3.1) and the runtime consumer (riir-ai Plan 642)
share ONE definition. The correctness axis is the **spec-match grid**
(in-module unit test `spec_match_grid_bit_identical_to_the_arena_rule`):
bit-identity (`.to_bits()`) against the arena's inline reference over the
full rank × fired-fraction cross product — that is the byte-identical
adoption criterion, and it passed (4/4 unit tests + the doc example).

## Results

| Gate | Axis | Result |
|---|---|---|
| G1 (spec-match) | comparator ≡ arena rule, bit-identical grid | **PASS** (4/4 tests, full 3×3×8×8 cross product) |
| G1 (stability) | 10⁶ shuffles of tied inputs — full tie scores exactly 0.5 both orders; all-equal table → first index under every permutation | **PASS** (0 violations; sink `1e6`) |
| G4 (alloc) | comparator + `_with_ticks` + champion scan over 200k mixed calls | **PASS** (0 allocations) |
| G2 (readout, informational — no bar) | ns/call | two_tier_score **1.01** · _with_ticks **1.01** · champion_index(16) **0.95** |

**Box state:** M3 Max (macOS 27.0.1), dev profile `cargo bench` (release
posture per the bench harness), AC power, shared box under concurrent
agent load (CPU ~82% at run time) — the G2 row is INFORMATIONAL by
design (no bar to flake under load); the load-bearing gates (G1/G4) are
load-insensitive by construction.

## Verdict

**GOAT (extraction class) — G1 + G4 PASS, G2 informational.** The
promotion is ADOPTION-GATED per Plan 631 T2.2: the feature promotes to
the consumers' default surface only after their benches land (sdk Plan
009 T4 / riir-ai Plan 642 G2). Until then `harness_evolution` stays
opt-in with zero default-feature surface change (verified: default
posture clippy clean).

## Scope notes (honest)

- `BestKeeper` (631 T1.2) and `MutationArms` (T1.3) are **NOT landed** —
  the ≥2-committed-consumers condition is not met (the arena generations
  driver keeps its own `GenerationLog`; riir-refine `EvolveRecorder` is
  not a committed consumer). They stay consumer-local per the plan's own
  conditional.
- The consumer adoption (sdk 009 T3.1) lands in the same wave; its
  byte-identical acceptance is the arena's unchanged G1 gate
  (`arena_gates::g1_two_full_tournaments_bit_identical`), which pins Elo
  bit-identity end to end.
