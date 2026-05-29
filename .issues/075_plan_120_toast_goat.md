# Issue 075: Plan 120 T5-T6 ToaST GOAT Proof

**Date:** 2026-05-29
**Plan:** 120
**Status:** DEFERRED
**Priority:** LOW
**Feature Gate:** toast

## Problem

Plan 120 Tasks 5-6 require a GOAT proof comparing ToaST compression against BPE baseline, plus implementation of the Rényi entropy efficiency metric.

## Tasks

- [ ] T5: GOAT proof — ToaST compression ratio vs BPE on representative corpus
- [ ] T6: Implement Rényi entropy efficiency metric for tokenization quality measurement
- [ ] Run comparative benchmarks: compression ratio, vocabulary utilization, downstream perplexity
- [ ] Document results with Rényi efficiency scores

## Context

ToaST (Tokenization-optimized Adaptive Subword Tokenizer) is designed to improve upon BPE by using adaptive subword segmentation. The Rényi entropy metric provides a principled information-theoretic measure of tokenization efficiency beyond simple compression ratio.

## Blockers

Needs a corpus pipeline to feed representative text through both ToaST and BPE tokenizers. The Rényi metric implementation requires access to token frequency distributions from real tokenized corpora.
