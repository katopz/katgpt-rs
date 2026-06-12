# Issue 002: ANE Concat-Tap Forward Kernels for Larger Models

**Source:** Research 223 — maderix/ANE Distillation
**Status:** Deferred
**Blocked On:** Multi-layer model integration (currently 1-layer microGPT)
**Priority:** Low

## What

Modify ANE forward pass to use concat-tap pattern: output not just the result but also intermediate activations (Q, K, V, xnorm, etc.) in a single dispatch. Eliminates redundant ANE calls for backward pass and KV cache warm.

## Why

- maderix/ANE saves all intermediates in single ANE dispatch during forward
- Reduces ANE dispatch count by 2-3× per layer
- Critical for multi-layer models (12 layers × 3 saved dispatches = 36 fewer dispatches)

## Blocker

- Our microGPT is 1-layer — only saves 2-3 dispatches total
- Not worth the complexity for current scale

## When to Unblock

- When katgpt-rs supports multi-layer models (n_layer > 4)
- When KV cache warm needs saved intermediates

## Tasks

- [ ] Extend MIL/kernel generation to concat outputs
- [ ] Modify forward pass to extract and cache intermediates
- [ ] Benchmark: single dispatch vs multiple dispatch for 12-layer model
- [ ] Verify intermediate accuracy (cosine ≥ 0.999 vs separate computation)
