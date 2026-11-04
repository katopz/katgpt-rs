# Bench 979: two-region probe-grammar recovery — the Research-650 §4.1 smallest step (NS-CSG × 649)

**Date:** 2026-11-05
**Status:** PASS — the pre-registered bar met on 3/3 pinned configs; the sweep's "recorded, not run" smallest consumer-free step is now RUN.
**Feature:** `probe_circuit_grammar` (katgpt-core, opt-in, no default consumer — unchanged)
**Research:** [`.research/650_Probe_Grammar_Consumer_Fusion_Sweep.md`](../.research/650_Probe_Grammar_Consumer_Fusion_Sweep.md) §4.1 (the NS-CSG × 649 NOTE row and its smallest step) over [`.research/649_Probe_Authored_Circuit_Grammar.md`](../.research/649_Probe_Authored_Circuit_Grammar.md) (the anchor primitive) and [Bench 978](978_probe_authored_circuit_grammar_poc.md) (the single-region PoC this extends).
**Test:** `crates/katgpt-core/tests/probe_circuit_grammar_poc.rs` `poc_p4_two_region_partition_recovery` (P4; P0–P3 byte-identical postures, 6/6 green)
**Pre-registered bar (Research 650 §4.1, quoted):** *"extend the Bench-978 injected-degeneracy MLP to TWO disjoint band-gates in two half-space regions; `classify` keys the region; the per-region pipeline must recover both k₀ exactly. Both constants exact, referee emits exact k on 3/3 pinned configs."*

## The claim this proves

The 649 pipeline parses ONE band-gate over ONE scalar gate variable (its own caveat 3). The NS-CSG fusion's lift is the REGION: partition the input space into disjoint regions, run the pipeline PER REGION, and the per-region records describe gates a single scalar band cannot. This bench runs that lift in miniature, consumer-free: two disjoint band-gates (distinct carrier column, band, and k₀ per region) in two half-space regions, a coverage-guaranteed region classifier keying the capture pools step-by-step, and the per-region parse → emit → combined-overlay repair. The real `BorelPartition` ships downstream (riir-engine `ns_csg.rs`); katgpt-rs stays consumer-free, so P4 carries the CONTRACT (disjoint + exhaustive assignment, every input lands in exactly one region, compiler-enforced by an exhaustive `match`) on a test-local half-space classifier.

## The pinned config set (constants pinned before the run)

`TwoRegionParams { split_x, row, a: (carrier, band, k₀), b: (carrier, band, k₀) }` — constraint set from the fixture's analytic shape (never tuned per run): distinct carrier columns from `{3,4,5}` (a shared column would sum the two injections into one entry and degenerate the record to a single word); both off-diagonal (≠ the shared gap row; twin value exactly 0); DISTINCT band grid cells of `{2..6}` (one scalar band cannot describe both — the region is the disambiguator, which is the lift under test); distinct k₀; `split_x ∈ [−3, 1]` keeps both start sides non-empty.

| config | split_x | shared row | region A (col, band, k₀) | region B (col, band, k₀) |
|---|---|---|---|---|
| TR_ALPHA | 0.0 | 1 | (4, 4, 8) | (3, 3, 6) |
| TR_BETA | 0.0 | 3 | (4, 3, 5) | (5, 5, 10) |
| TR_GAMMA | −1.0 | 5 | (3, 3, 6) | (4, 4, 9) |

Starts: per-region in-band grids (assigned THROUGH the classifier, never a hand split) + out-of-band marching starts (`|y|` beyond both bands — the crossing episodes that make step-level tagging load-bearing: an episode can oscillate in one region's pool and march through the other's) + the clean grid.

## Results — the bar, met on 3/3

Per-region blind parse (probe sees rollout taps + the W3 readout row only; ground truth asserted): exactly ONE candidate column clears the gate bar in each region pool (the other region's carrier is bit-exactly zero at every foreign-region step — contrast 0), drive col / band / slope-window all correct, digest-stable on re-capture.

| config | region | parsed (col, band) | slope (true −k₀) | referee emits |
|---|---|---|---|---|
| TR_ALPHA | A | (4, 4) ✓ | −7.5207 (−8) | **8 = k₀** ✓ |
| TR_ALPHA | B | (3, 3) ✓ | −5.6026 (−6) | **6 = k₀** ✓ |
| TR_BETA | A | (4, 3) ✓ | −4.6020 (−5) | **5 = k₀** ✓ |
| TR_BETA | B | (5, 5) ✓ | −9.4447 (−10) | **10 = k₀** ✓ |
| TR_GAMMA | A | (3, 3) ✓ | −5.6020 (−6) | **6 = k₀** ✓ |
| TR_GAMMA | B | (4, 4) ✓ | −8.5217 (−9) | **9 = k₀** ✓ |

Combined overlay `[spec_A, spec_B]`: obstacle 0.0 AND failure 0.0 on every start (in-band both regions + crossing + clean), every episode reaches the goal. Isolation: each single word alone repairs ITS region and leaves the other at failure 1.0; out-of-region single forwards bit-identical (foreign carrier mass exactly 0 → delta exactly 0). Artifact provenance carries the JOINED fn-grammar record (BLAKE3 over the two binding digests, region order fixed A then B) and round-trips the freeze envelope.

## Finding 1 — the region composition has a DEPENDENCY ORDER (the run's measured discovery)

The P4 run's first emission attempt failed `NoViableStrength` on region A — and the reason is structural, not a bug: region A is the LEFT region; its repaired episodes march right and ENTER region B, so **no single-A word can drive A's own authoring failure rate to 0 while B is broken**. Region B's repaired episodes march AWAY from A (toward the goal), so B referees standalone. The working order is **far region first** (the region repaired episodes exit INTO last), then the region whose repair flow crosses it — A's sweep composes B's found word into its referee via the documented caller-side primitive (`binding_to_spec`, ascending-first-accept over the composed overlay, same acceptance semantics as `emit_word`). P4 asserts the asymmetry on the exact constant: A's word ALONE leaves A's own authoring set failing (`> 0.0`), B's word alone referees clean.

**Generalization (honest scope):** in this fixture the order follows the reward gradient (episodes flow A → B → goal). A world with cyclic region flow (repair flow crossing a region that flows back) would need a JOINT sweep, not a sequence — the two-region DAG is the trivial case. Recorded so a future consumer doesn't read "far first" as a universal rule.

## Scoping notes (the honest halves)

- **Shared gap row:** the two regions share ONE vertical-decision row (ONE readout surface). `discover_gap_row` is argmax over a single row — a region-conditional gap row needs a per-region readout surface (out of scope for the smallest step; recorded in the test header too). The lift demonstrated is the region-conditional DRIVE/BAND/STRENGTH record.
- **Classifier posture:** the region is a fixture constant (`split_x`), given to the lane like the readout rows are; the lane never derives it. The NS-CSG fusion would supply the partition from BFCP machinery — that supply line is exactly what the consumer-side unblock triggers gate.
- **No latency claims:** a correctness PoC (dev profile, M3, other cargo jobs present); no timing numbers are asserted or quoted.

## What this unblocks (unchanged triggers, de-risked path)

The NS-CSG × 649 NOTE row (Research 650 §4.1) stays NOTE: its consumer-side unblock triggers are untouched — (a) the decstat outcome join unblinds `esc_voi`'s value legs; (b) a second armed suite (rethink issue-023 family) makes region-keyed gate fitting measurable. What changed: the region-conditional pipeline is no longer a hypothesis — per-region pools → per-region parse → exact per-region emission → composed repair is demonstrated end to end at the smallest scale, with the composition's dependency order already mapped. The whole-649-§8 sweep-level re-open trigger also stands: any documented weight-repairable degeneracy at a named consumer re-opens the family.
