# Bench 886 — differential anchor + fitted-anchor-table substrate GOAT (Issue 882 P0 + 883 P0 substrate)

**Status:** COMPLETE — G1 PASS · G2 PASS (0.031 µs axpy vs the 1 µs bar; full-pass 1.0049 vs the 1.02 bar) · G3 PASS (λ=0 bit-identical, pinned) · G4 PASS (0 steady-state allocs, both paths). Features `differential_anchor` + `fitted_anchor_tables` (katgpt-core) stay **OPT-IN** per the no-default-consumer rule — promotion rides a live consumer's GOAT (882's P0 rider: the healer rerank lane).

**Box:** 4090 workstation (i7-13700K, Windows 11, AC power, idle desktop — GPU idle at 20% compositing only, no compute consumers). Release profile. G2 via the shared interleaved `ab_median_ratio` (30 rounds) + `best_of_us` (µs over a 64-correction batch — a single 512-wide axpy is below timer resolution in release, the instrument's own loud-zero defence fired on the first draft).

## What was measured

The Issue 882 P0 primitive (`q̂ = q − λ·ā`, one axpy) and the Issue 883 P0 shared table-builder substrate (`StreamingMeanTable::observe` — the calibration-loop hot path both issues share):

- **G2a**: `correct_query_into` at d=512, batched ×64, black_box on result and arguments: **0.031 µs/call** (bar 1 µs — the O(d) axpy is invisible against the ~123 µs latent-KNN p50 it composes with).
- **G2b**: corrected vs plain full scoring pass (Q=64 queries × C=256 candidates × d=512): **median 1.0049** (min 0.881, max 1.079 across 30 interleaved rounds; bar 1.02 — the correction is 1/(1+C) ≈ 0.4% of the per-query work, measured at exactly that).
- **G4a**: corrected scoring pass with caller scratch: **0 allocs**.
- **G4b**: substrate `observe` loop (64 tracked + 1 tail, d=512): **0 allocs** — the 883 calibration path's hot loop is alloc-free by construction (rows allocated once at table build).
- **G4c**: in-place `correct_query`: **0 allocs**.

## G1 (quality, unit-gated in `differential_anchor::tests`)

The synthetic hub world (D=48, Q=24, 8 hubs): every query = private direction + shared generic direction; hubs carry only the generic direction (cos ≈ 0.905 vs the true match's ≈ 0.48). At λ=0 hubs win **0/24 recoveries… all 24 queries** (base = 0); grid-evaluated λ* = 0.55 recovers **24/24**, and the hubness statistic strictly decreases.

## Findings

1. **The mean-similarity distribution's skewness is INVARIANT under a mean-query anchor — measured, not assumed.** `mean_q[(q−λā)·d] = (1−λ/|m|)·mean_q[q·d]` — a pure scalar per candidate (ā IS the normalized mean query), and standardized skewness is scale-invariant, so that statistic cannot move at any λ. The G1 session's first gate asserted it and failed at EXACT equality (1.088733 vs 1.088733, every candidate's mean-sim scaled by the identical 2.525). **The measurable hubness axis under a mean-query anchor is the per-candidate top-1 WIN-COUNT distribution (the CSLS k-occurrence form)** — hubs concentrate wins at λ=0 (right-skewed), correction spreads them one-per-true-match (skewness strictly ↓, pinned). The invariant itself is now pinned as a test (`mean_sims` scale-uniformly + skew-equal) so nobody re-derives the "statistic that cannot move" as a future gate. Mean-CORPUS anchors do not have this invariant (their penalty term is not proportional to the mean-query dot).
2. **The smallest-λ tie-break matters on flat metrics**: on the hub world the top-1 metric plateaus across λ ∈ [~0.5, ~1.2]; `pick_lambda_by_eval` deterministically returns the plateau's LEFT edge (least correction that achieves the max), never the largest.
3. **The substrate's tail-lump R² is a strict lower bound, verified in both directions**: an internally-homogeneous tail makes the lumped partition exact (ρ unchanged); an internally-split tail (two true keys fused) reports ρ strictly below full tracking with SST exact — the honest go/no-go semantics for 883 P0's partial top-K calibration.

## Gates

- G1: `differential_anchor::tests::g1_synthetic_hub_world` (+ the invariant pin) — unit-gated, no RNG (orthogonal-coordinate fixture, deterministic across platforms).
- G2: this bench (`cargo test --release -p katgpt-core --features differential_anchor --test bench_886_differential_anchor_goat`).
- G3: `lambda_zero_is_bit_identical` (to_bits comparison incl. −0.0/1e-20/1e20 entries).
- G4: this bench, counting allocator.

## Posture

Opt-in (`differential_anchor`, `fitted_anchor_tables`); λ=0/default paths bit-identical; no default feature set changes. The 883 P0 calibration harness (riir-infer gemma taps) consumes `fitted_anchor_tables` next — the substrate's GOAT here is the P0 entry point's foundation, not a 883 quality claim (P0 is measurement-only by the issue's own law).
