# Issue 073: Plan 128 T8 Proof Sketch GOAT Benchmark

**Date:** 2026-05-29
**Plan:** 128
**Status:** DEFERRED
**Priority:** MEDIUM
**Feature Gate:** proof

## Problem

Plan 128 T8 requires a GOAT benchmark measuring convergence speedup from proof-sketch pruning. This needs real arena runs to produce meaningful measurements.

## Tasks

- [ ] Design convergence speedup benchmark: compare proof-sketch pruning vs baseline across multiple prompt types
- [ ] Run arena benchmarks measuring token efficiency (fewer tokens to reach target quality)
- [ ] Measure latency impact of proof-sketch overhead vs quality gain
- [ ] Document convergence curves and speedup factors

## Context

The proof pruner core implementation exists in `src/pruners/proof/`. Proof-sketch pruning uses reasoning structure to guide attention allocation, theoretically improving convergence by focusing computation on relevant token ranges. The pruner is functional but unbenchmarked at scale.

## Blockers

Needs real arena runs with sufficient diversity of reasoning tasks. Convergence speedup is only measurable with multi-step generation benchmarks (e.g., chain-of-thought, mathematical reasoning).
