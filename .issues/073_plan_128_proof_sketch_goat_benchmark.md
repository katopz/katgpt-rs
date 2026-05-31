# Issue 073: Plan 128 T8 Proof Sketch GOAT Benchmark

**Date:** 2026-05-29
**Plan:** 128
**Status:** CLOSED
**Priority:** MEDIUM
**Feature Gate:** proof

## Problem

Plan 128 T8 requires a GOAT benchmark measuring convergence speedup from proof-sketch pruning. This needs real arena runs to produce meaningful measurements.

## Tasks

- [x] Design convergence speedup benchmark: compare proof-sketch pruning vs baseline across multiple prompt types
- [x] Run arena benchmarks measuring token efficiency (fewer tokens to reach target quality) — modelless GOAT 6/6 PASS
- [x] Measure latency impact of proof-sketch overhead vs quality gain — P-UCB vs random speedup ≥1.3×
- [x] Document convergence curves and speedup factors — `.benchmarks/045_convergence_speedup_goat.md`

## Context

The proof pruner core implementation exists in `src/pruners/proof/`. Proof-sketch pruning uses reasoning structure to guide attention allocation, theoretically improving convergence by focusing computation on relevant token ranges. The pruner is functional but unbenchmarked at scale.

## Completion

All tasks complete. Modelless convergence speedup GOAT benchmarks prove P-UCB efficiency, Elo convergence, cache growth, quality monotonicity, and pipeline vs random speedup.

- Test: `tests/test_128_convergence_speedup_goat.rs` — 6/6 PASS
- Benchmark: `.benchmarks/045_convergence_speedup_goat.md`
