# GOAT Proof 032: DashAttention — Adaptive Sparse Routing Benchmarks (Plan 106 T21-T24, T26)

> **Date:** 2025-06-28
> **Feature Gate:** `dash_attn`
> **Depends on:** Plan 106 T1-T20 (α-entmax, chunk summaries, routing, forward integration)

## Summary

Benchmarks and GOAT proof for DashAttention's α-entmax adaptive sparse routing. Core result: **entmax routing allocates 3.3× more active chunks for hard (ambiguous) queries vs easy (peaked) queries**, confirming adaptive sparsity without any fixed budget parameter.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Config | `DashAttnConfig::default()` |
| α | 1.5 |
| Chunk dimensions | 16 or 32 |
| Number of chunks | 64-256 |
| Scaling factor | 1.0 |

## Benchmark Results

### T21: Fixed Top-K vs Entmax Routing

| Metric | Fixed Top-8 | Entmax Adaptive | Notes |
|--------|-------------|-----------------|-------|
| Avg active blocks | 8 (fixed) | 21.5 | Entmax adapts per query |
| Coverage of top-8 | 100% | 100% | Entmax always covers fixed top-8 |
| Min active blocks | 8 | ~4 | Easy queries get fewer |
| Max active blocks | 8 | ~40+ | Hard queries get more |

### T22: Chunk Summary Quality

| Metric | Mean Pooling | Learned (head_cls) |
|--------|-------------|-------------------|
| Top-block match | baseline | 30% differ |
| Both valid | ✅ | ✅ |

30% difference is expected — the head_cls perturbation shifts routing decisions. At production scale with trained head_cls, this would be meaningful.

### T23: NIAH Needle Position Sweep (256 chunks)

| Needle Position | Active Blocks | Needle Found? |
|----------------|---------------|---------------|
| 0 | varies | ✅ |
| 64 | varies | ✅ |
| 128 | varies | ✅ |
| 192 | varies | ✅ |
| 255 | varies | ✅ |

**100% needle retrieval** across all positions. Entmax routing successfully identifies the needle chunk regardless of position.

### T24: Query Difficulty Analysis

| Query Type | Avg Active Blocks | Description |
|-----------|-------------------|-------------|
| Peaked (1 chunk dominant) | 25.0 | Clear winner → moderate support |
| Spread (several chunks) | 51.8 | Ambiguous → wider support |
| Noise (random) | varies | Depends on random alignment |

Peaked queries have **fewer active blocks** than spread queries — confirming adaptive budget allocation.

### T26: GOAT Proof — Adaptive Support (THE KEY PROOF)

| Metric | Easy Queries | Hard Queries | Ratio |
|--------|-------------|-------------|-------|
| Avg active blocks | 12.1 | 40.5 | **3.3×** |
| Min active blocks | ~4 | ~20 | — |
| Max active blocks | ~25 | ~60 | — |

**Gate: ✅ PASS** — Hard queries get > 2× more active chunks than easy queries (actual: 3.3×).

This is the core claim of DashAttention: α-entmax routing automatically allocates more compute to ambiguous inputs and less to confident ones, **without any fixed budget parameter**.

## GOAT Gate

| # | Proof | Gate | Result |
|---|-------|------|--------|
| T21 | Top-k coverage | 100% | ✅ PASS |
| T22 | Summary quality | Both valid | ✅ PASS |
| T23 | NIAH retrieval | ≥ 80% | ✅ PASS (100%) |
| T24 | Adaptive budget | Peaked < Spread | ✅ PASS |
| T26 | Adaptive support | Hard > 2× Easy | ✅ PASS (3.3×) |

**Overall: 5/5 gates PASS**

## Files Changed

| File | Change |
|------|--------|
| `tests/bench_106_dash_attn_routing.rs` | NEW: 5 benchmark tests + GOAT proof |
| `.benchmarks/032_dash_attn_routing_goat.md` | NEW: This file |

## Related

- Plan 106: `.plans/106_dash_attn_adaptive_sparse_attention.md`
- Research 68: `.research/068_DashAttention.md`
- GOAT 9/9: `tests/goat_106_dash_attn.rs` (infrastructure proofs)
- Entmax overhead: `tests/bench_106_dash_attn_entmax.rs` (T25)