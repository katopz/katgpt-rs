# Bench 029 — CCE Primal-Dual Convergence (G2)

**Plan:** [295](../.plans/295_lp_cce_moderator_primitive.md) Phase 2, Task T2.4
**Research:** [274](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
**Paper:** [arxiv 2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas 2026, Theorem 6.1
**Date:** 2026-06-20
**Feature gate:** `cce_moderator` (default-off)
**Test binary:** `cargo test --features cce_moderator --test cce_convergence -- --nocapture`

---

## Setup

Emission-abatement discrete example (paper §8.2, simplified to 1-firm 4×4):

- **States** `N = 4`: market price signals `{Low, Med, High, Critical}`.
- **Actions** `A = 4`: abatement levels `{None, Light, Moderate, Heavy}`.
- **Cost matrix** `cost(s, a)`:

| | a=0 (None) | a=1 (Light) | a=2 (Moderate) | a=3 (Heavy) |
|---|---|---|---|---|
| **s=0** (Low) | 1.0 | 2.0 | 3.0 | 4.0 |
| **s=1** (Med) | 3.0 | 2.5 | 3.5 | 4.5 |
| **s=2** (High) | 6.0 | 4.0 | 3.0 | 5.0 |
| **s=3** (Critical) | 10.0 | 7.0 | 4.5 | 4.0 |

- **Deviation class** `|D| = 4`: constant deviations `{always-None, always-Light, always-Moderate, always-Heavy}`.
- **Moderator objective** `γ₀(ρ) = γ(ρ)` (firm's expected cost).

## LP Reference

`CceLp::solve` (BFS enumeration, `C(20, 5) = 15504` candidates):

- `ρ⋆ = δ_{(s=0, a=0)}` (Low price, no abatement) — the globally cheapest pair.
- `γ₀(ρ⋆) = 1.000000`
- `ER(ρ⋆) = 0.000000` (marginal CCE — always-None deviation ties).

## Primal-Dual Configuration

- `CcePrimalDual::new::<4, 4>().with_eta(0.05)`
- Initialization: `ρ⁰ = uniform`, `λ⁰ = 0`.
- Bregman potential: Euclidean (`ψ(ρ) = ½‖ρ‖²` → projected gradient).
- Steps: `N = 10⁴` (G2a/G2b) and `N = 3·10⁴` (G2c slope fit).

## Results

### G2a — Gap to LP optimum

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| `γ₀(ρ̄ᴺ)` | 1.000784 | — | — |
| `\|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)\|` | **0.000784** | `< 0.05` | **PASS** ✅ |

### G2b — CCE feasibility of averaged iterate

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| `ER(ρ̄ᴺ)` | **0.000034** | `≤ 0.05` | **PASS** ✅ |

### G2c — Empirical convergence rate

Sampled gaps at geometrically-spaced iteration counts:

| `n` | `\|γ₀(ρ̄ⁿ) − γ₀(ρ⋆)\|` |
|---|---|
| 100 | 0.079867 |
| 300 | 0.026622 |
| 1000 | 0.007987 |
| 3000 | 0.002662 |
| 10000 | 0.000799 |
| 30000 | 0.000266 |

Least-squares fit on `(log n, log gap)`:

- **Fitted slope: −1.0000**
- Paper Theorem 6.1 upper bound: `−0.5` (i.e., `gap(N) ≤ C · N⁻¹ᐟ²`).
- The empirical slope is **steeper** (−1.0 ≈ `O(N⁻¹)`) — faster than the
  worst-case bound. This is expected for a well-conditioned problem with a
  unique vertex optimum; the `O(N⁻¹ᐟ²)` rate is a worst-case guarantee, not
  a lower bound. **PASS** ✅ (slope ≤ −0.3, satisfies the upper bound).

## Verdict

**G2 PASS** — all three sub-gates green. The `CcePrimalDual` iterator
converges to the `CceLp` optimum on the emission-abatement example, with
the averaged iterate satisfying the CCE constraint within Slater tolerance.

## Caveats

1. **Single-player model.** The test uses a 1-firm emission problem. Multi-player
   CCE (e.g., chicken with both players' deviation constraints) requires
   extending the deviation class to cover all players — deferred to riir-ai
   Plan 325.
2. **Faster-than-worst-case rate.** The empirical `O(N⁻¹)` convergence on this
   problem is faster than the paper's `O(N⁻¹ᐟ²)` worst-case bound. The worst-case
   rate would manifest on adversarial payoff tensors; we have not constructed
   such a stress test.
3. **No dynamics.** The paper's full formulation includes a transition kernel
   (`occupation measure flow = π_recommendation marginal`). This benchmark
   uses a 1-shot game without dynamics — the LP treats the state distribution
   as free. Adding dynamics is a Plan 325 follow-up.

## Cross-References

- Plan: [`katgpt-rs/.plans/295_lp_cce_moderator_primitive.md`](../.plans/295_lp_cce_moderator_primitive.md) Phase 2
- Research: [`katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md`](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
- Test source: [`katgpt-rs/tests/cce_convergence.rs`](../tests/cce_convergence.rs)
- GOAT gate aggregation: [`katgpt-rs/.benchmarks/029_cce_moderator_goat.md`](029_cce_moderator_goat.md) (Phase 4)
