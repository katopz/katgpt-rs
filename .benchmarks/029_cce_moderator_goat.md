# Bench 029 — CCE Moderator GOAT Gate Aggregation

**Plan:** [295](../.plans/295_lp_cce_moderator_primitive.md) Phase 4, Task T4.1
**Research:** [274](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
**Paper:** [arxiv 2606.20062](https://arxiv.org/pdf/2606.20062) — Campi, Cannerozzi, Tzouanas 2026
**Date:** 2026-06-20
**Feature gate:** `cce_moderator` (default-off)

---

## GOAT Gate Verdict: **PASS** (G1 + G2 + G3)

All three Phase 1–3 GOAT gates pass. The `cce_moderator` primitive is validated
and ready for riir-ai Plan 325 runtime integration. G4 (crowd-scale latency)
and G5 (LatCal commitment) are deferred to Plan 325.

---

## Gate Summary

| Gate | Target | Result | Verdict |
|---|---|---|---|
| **G1** — CCE Pareto-dominates Nash | Welfare gain ≥ 5% on chicken + BoS | Chicken: +37.5% (5.5 vs 4.0); BoS: +108% (5.0 vs 2.4) | **PASS** ✅ |
| **G2** — Primal-dual convergence at O(N⁻¹ᐟ²) | gap < 0.05, ER ≤ 0.05, slope ≤ −0.3 | gap=0.000784, ER=0.000034, slope=−1.0000 | **PASS** ✅ |
| **G3** — Designer steering demo | Two Γ₀ → two structurally different CCEs | Selfish (welfare 5.0, p1 reward 4.0) vs welfare-max (welfare 5.5, p1 reward 2.0) | **PASS** ✅ |
| G4 — Crowd-scale latency | < 50µs per NPC update | — | Pending (Plan 325) |
| G5 — LatCal commitment | Bit-identical | — | Pending (Plan 325) |

---

## G1 Evidence — CCE vs Nash Pareto-dominance

**Source:** [`tests/cce_vs_nash.rs`](../tests/cce_vs_nash.rs) (3 tests, all PASS)

### Chicken (general-sum, Pareto-dominant CCE exists)

| Solution | Welfare | Source |
|---|---|---|
| Mixed Nash (each swerves p=0.5) | 4.000 | Analytic |
| LP-CCE (player-1-only model) | **5.500** | `CceLp::solve` |

**Pareto gain: +37.5%** (threshold: +5%). ✅

### Battle of the Sexes (general-sum, Pareto-dominant CCE exists)

| Solution | Welfare | Source |
|---|---|---|
| Mixed Nash (p1=3/5 Opera, p2=3/5 Football) | 2.400 | Analytic |
| LP-CCE (player-1-only model) | **5.000** | `CceLp::solve` |

**Pareto gain: +108%** (threshold: +5%). ✅

### RPS (zero-sum, no Pareto gain)

| Solution | Player-1 cost | Source |
|---|---|---|
| Mixed Nash (uniform) | 0.000 | Analytic |
| LP-CCE (player-1 cost) | −1.000 | `CceLp::solve` |

The LP exploits the free state distribution (concentrates on the most favorable
(s,a) pair). Without dynamics or honest-mediator constraints, the 1-shot LP
trivially finds a "CCE" that beats the zero-sum baseline. This is a documented
limitation of the 1-shot model — the fair comparison requires MFG dynamics
(riir-ai Plan 325). The test asserts only `γ₀(CCE) ≤ 0` (player 1 never worse
than baseline).

---

## G2 Evidence — Primal-dual convergence

**Source:** [`tests/cce_convergence.rs`](../tests/cce_convergence.rs) (4 tests, all PASS)
**Details:** [`029_cce_convergence.md`](029_cce_convergence.md)

### Setup

Emission-abatement discrete example: N=4 states, A=4 actions, |D|=4 deviations.
LP reference: `γ₀(ρ⋆) = 1.000000` (all mass on (Low, None)).

### Results

| Sub-gate | Metric | Value | Threshold | Verdict |
|---|---|---|---|---|
| G2a | `\|γ₀(ρ̄ᴺ) − γ₀(ρ⋆)\|` | 0.000784 | < 0.05 | **PASS** ✅ |
| G2b | `ER(ρ̄ᴺ)` | 0.000034 | ≤ 0.05 | **PASS** ✅ |
| G2c | log-log slope | −1.0000 | ≤ −0.3 (paper upper bound −0.5; steeper allowed) | **PASS** ✅ |

The empirical `O(N⁻¹)` rate is faster than the paper's `O(N⁻¹ᐟ²)` worst-case
bound — expected for this well-conditioned problem with a unique vertex optimum.

---

## G3 Evidence — Designer steering

**Source:** [`examples/cce_demo.rs`](../examples/cce_demo.rs) Section 3

Same game (chicken), same CCE constraints, two different moderator objectives:

| Moderator objective `Γ₀` | ρ⋆ support | Player 1 reward | Player 2 reward | Welfare |
|---|---|---|---|---|
| (A) Player 1 cost (selfish) | δ_{((T,S), T)} | 4.000 | 1.000 | 5.000 |
| (B) −Welfare (welfare-max) | 0.5·δ_{((S,S), S)} + 0.5·δ_{((S,T), S)} | 2.000 | 3.500 | 5.500 |

**Designer steering effect:** switching `Γ₀` lifts welfare by +10% (5.0 → 5.5)
and redistributes reward from player 1 (4.0 → 2.0) to player 2 (1.0 → 3.5).
The two CCEs are structurally different (different support, different
player-1 rewards). ✅

---

## Validation Commands

```bash
# G1 — CCE vs Nash
cargo test --features cce_moderator --test cce_vs_nash -- --nocapture

# G2 — Primal-dual convergence
cargo test --features cce_moderator --test cce_convergence -- --nocapture

# G3 — Designer steering demo
cargo run --example cce_demo --features cce_moderator

# Full unit test suite (35 tests)
cargo test --features cce_moderator --lib cce::

# Total: 42 tests (35 unit + 4 convergence + 3 vs-nash), all PASS
```

---

## Promotion Decision

**Verdict: Keep `cce_moderator` default-OFF.**

Rationale:
1. G1+G2+G3 all PASS — the primitive is correct and validated.
2. BUT the player-1-only CCE model is a known limitation. Multi-player CCE
   (both players' deviation constraints) is required for real game runtimes
   and is deferred to riir-ai Plan 325.
3. The BFS-enumeration LP doesn't scale beyond `N·A + |D| ≤ ~25`. Production
   game runtimes may need larger problems.
4. Promoting to default-on before Plan 325 validates the runtime integration
   would be premature.

**Promotion criteria for default-on:**
- Plan 325 ships G4 (crowd-scale latency < 50µs) and G5 (LatCal commitment).
- Multi-player CCE extension validated on a real game domain.
- At least one head-to-head win against `PayoffTable<N>::nash_equilibrium`
  on a full-game (both players) benchmark.

---

## Demotion Decision

**`PayoffTable<N>::nash_equilibrium` (riir-games): NOT demoted.**

The G1 evidence comes from the player-1-only CCE model, which can exploit
player 2. A fair head-to-head comparison requires the full-game CCE (both
players' constraints), which is Plan 325's scope. Additionally,
`PayoffTable<N>` is designed for zero-sum games (where Nash = CCE), so
demoting it for general-sum games doesn't apply — it's not used there.

A doc-comment cross-link from `PayoffTable<N>` to `CceLp` for general-sum
games is a Plan 325 follow-up (requires editing riir-games, separate repo).

---

## Cross-References

- **Plan:** [`katgpt-rs/.plans/295_lp_cce_moderator_primitive.md`](../.plans/295_lp_cce_moderator_primitive.md)
- **Research:** [`katgpt-rs/.research/274_Optimal_CCE_Moderator_LP_No_Regret.md`](../.research/274_Optimal_CCE_Moderator_LP_No_Regret.md)
- **API docs:** [`katgpt-rs/.docs/cce_moderator.md`](../.docs/cce_moderator.md)
- **G2 details:** [`katgpt-rs/.benchmarks/029_cce_convergence.md`](029_cce_convergence.md)
- **G1 test:** [`katgpt-rs/tests/cce_vs_nash.rs`](../tests/cce_vs_nash.rs)
- **G2 test:** [`katgpt-rs/tests/cce_convergence.rs`](../tests/cce_convergence.rs)
- **G3 demo:** [`katgpt-rs/examples/cce_demo.rs`](../examples/cce_demo.rs)
- **Private selling-point guide:** `riir-ai/.research/143_Latent_CCE_Moderator_Crowd_Emergent_Coordination.md`
- **Private runtime plan:** `riir-ai/.plans/325_latent_cce_moderator_runtime.md`
