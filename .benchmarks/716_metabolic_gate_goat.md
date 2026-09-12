# Bench 716 — Metabolic Gate GOAT gate (Research 550 / riir-ai Plan 585 T1.3+T1.4)

**Date:** 2026-09-12
**Primitive:** `katgpt-core` feature `metabolic_gate` — energy-coupled compute gating: `MetabolicGate::depth_factor` (σ((stock−base)/scale) compute-depth multiplier over a metabolic stock), `defector_starves` (Theorem-1 starvation law `2ε < L(1+(1−α)δ)`), `metabolic_drag_threshold` (Lemma-1 break-even inefficiency `K(ε,L)`, 32-iter bisection on `ln(2ε/(2ε−L(1+K))) = (1+K)·ln(ε/(ε−L))`), `steal_lossy` (conservation-exact lossy transfer, system destroys `(1−α)δ`), `execution_share` (lottery-share scheduling, both-zero → ½). Source: Jha, Cicala, Agüera y Arcas, Richards, Jaques, Kleiman-Weiner, Niklasson, "Tapes Together Strong: The Co-evolution of Computation and Cooperation" [arXiv:2609.10817].
**Files:** `src/metabolic_gate.rs`, `benches/bench_716_metabolic_gate_goat.rs`.
**Machine:** M3 Max (Apple silicon), `cargo bench` (release). **Ambient load honestly noted:** a sibling single-core precompute measurement + a Lean `lake` build were running (box load ~8/16 falling during the run); all figures are medians over ≥1000 timed batches — the robust statistic under that load class.

**Status: GOAT G1/G2/G3/G4 ALL PASS — STAYS OPT-IN** (the no-default-consumer rule; the riir-ai Plan 585 consumer re-gates at Phase 4 before any promotion decision).

---

## T0.2 — The recorded regime tuple

`ε = 12` (per-NPC per-tick energy grant), `L = 10` (spawn cost), `α = 0.3` (theft retention), `δ = 3` (per-theft transfer):

- Cooperator viability: `2ε = 24 > 2L = 20` ✓ (two cooperators can each replicate)
- Defector starvation: `24 < L(1+(1−α)δ) = 10 × 3.1 = 31` ✓ (Theorem 1)
- The regime is non-empty iff `(1−α)δ > 1` — here `0.7 × 3 = 2.1 > 1` ✓

## G1 — Correctness (design-law identities)

| Check | Result | Target |
|---|---|---|
| T0.2 tuple: viable + starving | both true | ✅ |
| K-solver residual \|lhs−rhs\| at returned K (4 regime pairs: 12/10, 20/10, 100/10, 12/11) | worst **6.7e-6** | < 1e-4 ✅ |
| Coherent story: `K(12,10) = 1.3654 < (1−α)δ = 2.1` | true | ✅ (the starvation tuple also sits above the Lemma-1 drag break-even) |
| `steal_lossy` conservation identity (binary-fraction α, exact f32) | `thief_gain + destroyed == victim_loss` | ✅ |
| `depth_factor` σ(0) anchor at `stock == e_base` | exactly 0.5 | ✅ |
| `execution_share(0,0)` → ½ (paper C.1) | 0.5 | ✅ |
| Unit tests (monotonicity, α→1 zero-sum recovery, δ=0 viability, exact-boundary `2ε == L(1+(1−α)δ)` → false, NaN-never-fires guards, partial-transfer clamp) | 6/6 | ✅ |

## G2 — Latency

| Call | p50 | Target |
|---|---|---|
| `depth_factor` | **3 ns** | < 10 ns ✅ |
| `execution_share` | **0 ns** (batched, sub-ns per call) | < 10 ns ✅ |
| `metabolic_drag_threshold` (32 fixed bisection iters, one log pair each) | **412 ns** | < 2000 ns ✅ |

## G3 — Feature isolation

- `cargo check -p katgpt-core --no-default-features`: clean (the module compiles away; the one pre-existing `special_fn::ln_gamma` dead-code warning belongs to the bmr feature family, not this primitive).
- `cargo clippy -p katgpt-core --features metabolic_gate --all-targets`: 0 warnings.
- Default-feature lib tests: unchanged (the module is absent).

## G4 — Zero-alloc hot paths

CountingAllocator over 1000 calls each: `depth_factor` **0**, `steal_lossy` **0**, `metabolic_drag_threshold` **0** — the K-solver is loop-only arithmetic, no heap by construction.

---

## Ledger

- Compute-budgeting family / **metabolic stock axis** — the sibling of `gain_cost_halt` (utility flow axis). Signal-diff: gain_cost_halt consumes (gain, cost) per decision; metabolic_gate consumes a persistent stock + design-law constants.
- Consumer: riir-ai Plan 585 (thermal-LOD metabolic axis Phase 2, swarm energy commons Phase 3). GOAT-audit re-run fires at Phase 2 wiring (consumption time).
- No UQ surface — no conformal floor required (Research 550 §4).
