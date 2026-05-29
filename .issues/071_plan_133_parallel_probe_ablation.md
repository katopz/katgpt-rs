# Issue 071: Plan 133 T4 Parallel Probe Ablation Benchmark

**Date:** 2026-05-29
**Plan:** 133
**Status:** OPEN
**Priority:** MEDIUM
**Feature Gate:** speculative

## Problem

Plan 133 T4 requires an ablation benchmark that measures accuracy and token impact when each parallel probe component is removed individually. The benchmark results file `.benchmarks/023_parallel_probe_goat.md` needs to be created with real measurements.

## Tasks

- [ ] Design ablation matrix: remove each component (draft model, tree scorer, early exit, etc.) and measure acceptance rate + latency
- [ ] Run ablation benchmarks on real hardware with meaningful sequence lengths
- [ ] Record accuracy metrics (acceptance rate, speculation accuracy) per ablation
- [ ] Record token impact (tokens per speculation round, throughput) per ablation
- [x] Benchmark file `.benchmarks/023_parallel_probe_goat.md` updated — 26/26 unit tests PASS

## Context

The core parallel probe implementation exists in `src/speculative/parallel_probe.rs`. The speculative decoding framework with draft model scoring and tree-based verification is functional. This task requires running controlled experiments rather than new code.

## Blockers

Needs riir-gpu speculative decode pipeline for real inference ablation.
