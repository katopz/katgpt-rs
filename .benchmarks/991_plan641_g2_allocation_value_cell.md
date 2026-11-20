# Bench 991 — Plan 641 G2: the epiplexity allocation-value cell

**Status:** VERDICT — G2 **FAIL** (the pre-registered falsifier fired; the primitive
stays opt-in, the consumer-ordering promotion is refused).

**Posture:** `cargo test --release -p katgpt-core --features estimators --test
epiplexity_gain_g2_bandit -- --ignored --nocapture` · **PROVENANCE:** power=AC Power
load=2.64 swap=0.00M canary=skipped powermode=2(high) ane=ok
(`../riir-reflex/scripts/bench_preflight.sh`, m3, 2026-11-20). Box-state law carried.

**Harness:** `crates/katgpt-core/tests/epiplexity_gain_g2_bandit.rs` (whole-file
`estimators`-gated; heavy arms `#[ignore]`d; 12 seeds × 200 rounds × chunk 8,
pick-based 4-domain bandit, 8-round rotation warmup identical for every arm, refit
every 8 rounds, dispersion-tempered sigmoid weight policy, momentum 0.75, exploration
floor 0.3, LP floor 0.25; paired LB95 = mean − t₀.₉₅,df=11·sd/√n).

## Main cell — the module's own shared-curve family

`gap(n) = ε + β·N^−α` over `N = Σ_k γ_k·(n_k+1)`; γ = [0.5, 0.3, 0.15, 0.05],
ε = 0.03, β = 0.7, α = 0.7 — recovery guaranteed by construction, the graded content
is the allocation value.

| arm | mean U | LB95(Δ vs ours) | verdict |
|---|---|---|---|
| ours (fitted) | 0.03996 | — | — |
| uniform | 0.04046 | **+0.00045** | ours better (PASS) |
| greedy-exp (η=4, max-norm) | 0.03869 | −0.00147 | tie (reported, non-gating) |
| **LP-slope — the falsifier** | 0.03919 | **−0.00093** | **FAIL (LB95 < 0)** |

Refit wall 1.416 s ≤ 12 s budget (PASS, release posture; debug posture measured
~15 s, disclosed, non-gating).

## Misspecification fixtures (additive per-domain worlds; bar: ours no worse than LP)

| world | U ours | U LP | mean(LP−ours) | lower95 | δ | verdict |
|---|---|---|---|---|---|---|
| exp-decay | 0.04600 | 0.04600 | −0.00000 | −0.00001 | 0.00046 | PASS |
| plateau-drop | 0.15400 | 0.15400 | 0.00000 | 0.00000 | 0.00154 | PASS |

## Root-cause evidence (offline replays, disclosed)

1. **Cross-arm γ is noise-dominated at the decision window**: the grid's winning
   (α, ε) cell lands off the truth (α 0.322 vs 0.7) with γ anti-correlated to the
   planted mix; a direct least-squares fit at the truth cell recovers
   (α 0.69, β 0.67, ε 0.03) — the family is recoverable, the robust-location
   grid+CD search does not find it on this data.
2. **The census demotes to Sparse at W ≈ 40** (eff10 < 3); the sparse per-domain
   gains rank correctly (d0 ≥ d1 > d2 ≈ d3) but weakly (≈2:1 spread) — the weight
   policy cannot concentrate past p(d0) ≈ 0.5, LP's own steady state.
3. **Per-pick slope attribution with an honest floor is near-optimal on stationary
   bandits**; the LP-floor sweep 0.05–0.40 never flips the sign. The fit's refit
   cadence + momentum are pure lag cost on a stationary world.

## Verdict + consequence

G2 **FAIL** per the plan's falsifier ("cannot beat the raw-slope arm with LB95 > 0 →
the fitting machinery is unjustified and the primitive demotes"). The primitive stays
opt-in under `estimators`; Plan 392's mining-queue gain-column promotion is REFUSED;
the queue order stays incumbent. Re-open bar: a world class where fitted shape
knowledge beats per-pick attribution (e.g. strongly non-stationary domains), or a
gain read that beats the sparse ranking's spread.
