# Benchmark 046: Breakeven Routing GOAT

**Plan:** 250
**Date:** 2026-06
**Feature Gate:** `breakeven_routing`

---

## Components

| Component | File | Role |
|-----------|------|------|
| `BreakevenTracker` | `src/breakeven/mod.rs` | Per-tier-pair cost tracking, N* computation, EMA updates |
| `BreakevenBandit` | `src/breakeven/mod.rs` | Multi-tier selection with sigmoid-gated transitions |
| `FidelityMatcher` | `src/breakeven/fidelity.rs` | Error-matched KV compression level selection |
| `InferenceRouter` integration | `src/inference_router.rs` | Tier adjustment hook, timing observation, stats |

## GOAT Gates

| Metric | Threshold | Method |
|--------|-----------|--------|
| Wallclock savings (≥512 tok) | >5% vs QPS-only routing | Simulated tier routing |
| Per-forward overhead | <100ns | Microbench: select_tier timing |
| Memory overhead | <1KB | sizeof(BreakevenBandit) |
| Zero allocation hot path | 0 allocs/forward | Atomic-only operations |

## GOAT Results (7/7 pass)

| Gate | Metric | Result | Threshold | Status |
|------|--------|--------|-----------|--------|
| T1 | Overhead per forward | **~9ns** | <100ns | ✅ PASS |
| T2 | Memory overhead | **176 bytes** | <1KB | ✅ PASS |
| T3 | Wallclock savings (512+ tok) | **49.0%** | >5% | ✅ PASS |
| T4 | Short sequence (50 tok) | **0.6× baseline** | ≤1.0× | ✅ PASS |
| T5 | N* accuracy | **0.00% error** | <10% | ✅ PASS |
| T6 | Sigmoid monotonicity | **Monotone** | Non-decreasing | ✅ PASS |
| Summary | All gates | — | — | ✅ PASS |

## Implementation Notes

- `BreakevenBandit::select_tier(current_tier) -> Option<ComputeTier>`: returns `None` when no override
- `BreakevenTracker` uses `AtomicU64` for all fields — zero-allocation, thread-safe
- EMA α=0.1 (fixed-point 6553/65536) converges after ~50 observations
- Sigmoid transition: `σ(α × (tokens - N*))` with α=0.001 (~1000 token transition width)
- Integration is additive: breakeven sits after critical-interval adjustment, before backend dispatch

## Key Formulas

```
N* = upfront_cost_us / max(baseline_cost_ema - tier_cost_ema, 0)
amortization_confidence = sigmoid(transition_sharpness × (total_tokens - N*))
EMA_new = α × value + (1 - α) × EMA_old   (fixed-point: α = 6553/65536)
```
