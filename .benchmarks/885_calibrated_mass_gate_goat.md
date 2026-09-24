# Bench 885 — calibrated-mass router gate GOAT: `gate_sigmoid_topk_mass_into` vs the incumbent (Issue 880)

**Status:** COMPLETE — G1 PASS · **G2 FAIL against the issue's ≤ 1.05× bar (by construction, see Finding 2)** · G3 PASS · G4 PASS. Feature `calibrated_mass_gate` (katgpt-spectral + root passthrough) stays **OPT-IN**: the router has no production consumer, so there is no gain to promote on, and the perf half of the gate does not hold.

**Box:** M3 Max (macOS 26.6.2), AC power, 88% memory free, load average 4.8–9.9 (concurrent agent sessions; no sibling release builds during the runs). Release profile. Timing via the shared interleaved `ab_median_ratio` (median of per-round `b/a` ratios, 31 rounds). Three runs; the table is run 1, and runs 2–3 are the range column.

## What was measured

`katgpt_spectral::manifold_power_iter_router::gate_sigmoid_topk_mass_into` **composes** the incumbent's logits and exact-k selection sort with an `exact_mass_admit_into` τ bisection, so that `Σᵢ σ((zᵢ − τ)/T) = k`. The arms:

- **a (incumbent):** `gate_sigmoid_topk_into`, with per-expert independent `fast_sigmoid` weights and uncalibrated mass.
- **b (calibrated):** `gate_sigmoid_topk_mass_into`, with the same index contract and weights σ(z − τ) at T = 1.

Fixture: d = 16, β = 1.3, router rows and inputs uniform in [−1, 1), and a 16-vector input pool cycled per iteration so neither arm can be hoisted.

## Results

| regime | b/a median (run 1) | runs 2–3 | a ns/call | b ns/call | \|Σw − k\| incumbent | \|Σw − k\| calibrated |
|---|---|---|---|---|---|---|
| hot N=64 k=4 | **11.02×** | 11.55× · 10.88× | 516 | 5,614 | 29.8 | 8.9e-8 |
| game N=256 k=8 | **6.57×** | 6.47× · 6.48× | 3,195 | 20,845 | 127.7 | 2.3e-7 |
| Bench-884 N=1000 k=100 | **1.75×** | 1.72× · 1.75× | 109,909 | 192,023 | 409.1 | 8.6e-7 |

## Findings

1. **Calibration is real and exact.** The calibrated gate holds |Σm − k| below 1e-6 in every regime. The incumbent's weights total about N/2 whatever k is (33.8 at N = 64, 135.7 at N = 256, 509 at N = 1000), because an independent sigmoid gives every expert near 0.5 weight on this fixture. It is not a budget.
2. **The ≤ 1.05× bar is unreachable for ANY calibrated gate, not just this implementation.** Bench 884's 0.906× compared the two operators as *alternatives* at k = N/10, where the incumbent's O(k·N) selection sort dominates. This upgrade has to keep the selection sort, because it is the index contract, and then add the calibration on top. Calibration needs several extra sigmoid passes over N. One pass costs about as much as the incumbent does in total (N dots plus one sigmoid pass). So the ratio has a floor well above 1.05× at game-scale k. The measured gap shrinks toward the Bench-884 regime (11× → 6.6× → 1.75×) exactly as the sort's share of the incumbent grows.
3. **The lever, if a consumer ever needs it cheaper, is the solver, not the gate.** The bisection runs about 32 passes to reach `MASS_TOL_REL = 1e-9·n`. Mass(τ) is smooth with a known derivative, Σ m(1 − m)/T, so a safeguarded Newton solve would need roughly 5–8 passes, a 4–6× cut to the calibration term. That alone still leaves the hot regime at about 3×, so it does not change the G2 verdict. It belongs in `katgpt-core::exact_mass_admit`, where it would have to re-earn that module's bit-exact shift-invariance pins, and only once a consumer shows the calibrated mass buys something. It is recorded here and not filed, because no consumer exists.

## Gates

- **G1:** PASS (katgpt-spectral unit suite, feature `calibrated_mass_gate`):
  - t15: selection is index- and order-identical to the incumbent, over 24 seeded fixtures across the three regimes.
  - t16: sum-to-k within 1e-3 + 2e-7·N, masses in [0, 1], masses non-increasing along the returned order, and every selected mass ≥ every unselected one.
  - t17: the documented **inverse of t06**. Perturbing one expert leaves the other logits bit-identical but moves τ, and with it their masses.
  - t18: the weights are exactly σ(z − τ), and τ ≈ 0 when k ≈ Σσ(z).
  - t19: budget extremes (k = 0 → zero mass and kk = 0; k > N → unit mass and every index).
  - The bench adds a spot calibration assert in every regime.
- **G2:** FAIL against the issue's ≤ 1.05× bar; see the table. The ratio is recorded rather than asserted. The only hard assertion is a gross de-optimization ceiling at 200×.
- **G3:** PASS. The incumbent was refactored onto the shared `router_logits_into` and `select_topk_desc_into` helpers, and t14 pins it **bit-identical** to its pre-refactor body across 40 fixtures, including a saturating β = 400 so that sigmoid ties exercise the sort's tie order. A mutation probe (`>` → `>=` in the shared sort) turns t14 red. All 14 pre-existing router tests pass unchanged, and `cargo clippy -D warnings` is clean with the feature both off and on.
- **G4:** PASS. 2,000 steady-state `_into` calls at N = 256 made **0 allocations**, measured with the canary-checked per-thread counting allocator.

## Verdict

The calibrated-mass gate ships **opt-in** and is not promoted. What it gives is accuracy: the weights become a real budget, which the incumbent's are not. What it costs is latency: 1.75× to 11× the incumbent, with no regime at the issue's ≤ 1.05× bar. A consumer that needs exact mass can now call a separately named, zero-alloc, tested gate instead of wiring `exact_mass_admit` itself. The promotion question needs that consumer's own GOAT.

## Reproduce

```bash
cargo test -p katgpt-spectral --features calibrated_mass_gate --lib manifold_power_iter_router
cargo test --release --features calibrated_mass_gate \
  --test bench_885_calibrated_mass_gate_goat -- --nocapture --test-threads=1
```

Cross-refs: [Bench 884](884_exact_mass_admit_goat.md) (the operator's own GOAT; this bench corrects how its Finding 2 transfers to the upgrade) · resolution record: HISTORY.md §Issue 880.
