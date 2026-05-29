# Issue 072: Plan 132 T24-T26 Committee Boost GOAT Proof

**Date:** 2026-05-29
**Plan:** 132
**Status:** OPEN
**Priority:** MEDIUM
**Feature Gate:** committee_boost

## Problem

Plan 132 Tasks 24-26 require a GOAT proof benchmark for the committee boost pruner, a benchmark results file, and a README update documenting the feature.

## Tasks

- [ ] T24: GOAT proof benchmark on Bomber arena — oracle-gap recovery, debiased comparison, budget sizing
- [x] T25: Benchmark results file `.benchmarks/020_committee_boost_goat.md` — 68/68 unit tests PASS
- [ ] T26: Update README.md with committee boost documentation section

## Context

The committee boost pruner implementation exists in `src/pruners/committee_boost/`. This is a multi-expert attention pruning strategy that uses committee voting to select which attention heads/patterns to retain. The core pruning logic is complete; what remains is verification and documentation.

## Blockers

Needs riir-games Bomber arena for oracle-gap measurement.
