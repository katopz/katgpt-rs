# Bench 952 — Issue 952 T4: bet-search GOAT (calculation brain over authored bet compositions)

**Status:** DONE 2026-10-22 — GOAT PASS for `bet_search` (exhaustive search + floor demotion) → **promoted to katgpt-extract default**; `bet_search_puct` **demoted to opt-in** (does not beat exhaustive at equal budget).

**Target:** `crates/katgpt-extract/tests/bench_952_bet_search_goat.rs` (both arms `#[ignore]`'d)

```bash
cargo test -p katgpt-extract --features bet_search_puct --release \
    --test bench_952_bet_search_goat -- --ignored --nocapture --test-threads=1
```

**Box state (M3 Max, 64 GiB):** AC power, battery 100% charged; load average 13.9/13.1/11.8 (several sibling sessions); swap 17.6 of 19.0 GiB used, `memory_pressure` 84% free. Wall-clock (G2) figures are from a LOADED box — the quality tables are deterministic (fixed seeds, bit-identical across three runs this session) and box-independent.

## Design

Five synthetic estimate worlds, each built so a DIFFERENT authored bet is right: `Ratio` (value = 3·x0 · (1+0.08g)), `Plane` (2·x0 + 5·x1 + g), `Smooth` (10 + 4·sin(0.6·x0) + 0.3g), `Flat` (50 + 2g), `Drift` (20 + 0.5·t + 0.5g; held-out queries continue the clock — distribution shift no bet extrapolates). Comps size n ∈ {3, 5, 10, 30, 60}, 100 seeded tasks per cell, 50 HELD-OUT queries per task (never comps). α = 0.2 (80% interval).

Every bet arm runs through the same `BetSearchHead` (same selection/calibration split, same calibration-split conformal half-width), so arms differ only in how the composition is chosen. Arms: **floor** (conformal-naive, `Fixed { mask: 0 }`) · **incumb** (the incumbent `estimate::EstimateHead`, kNN k=3, jackknife+ below 8 comps) · **best1** (best single bet = 7 singletons) · **knn5** (plain kNN, no search) · **full** (all 63 compositions) · **puctB** / **exh@EB** (PUCT at budget B vs exhaustive at the SAME number of distinct compositions PUCT scored on that task) · **served** (`BetSearchConfig::default()` — full exhaustive + the floor demotion: the shipped posture). Raw arms run with `demote: false`.

Skill = `(score_floor − score_arm) / score_floor`, averaged per task (scale-free). Cell format below: `skill coverage% %tasks-beating-floor`.

## Results — pooled over the five worlds

```text
## pooled over worlds — mean skill vs floor (coverage, % tasks beating floor)
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   66%   0w | +0.366   40%  78w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w
   5 | +0.000   78%   0w | +0.419   41%  80w | +0.280   61%  70w | +0.309   61%  76w | +0.296   61%  71w | +0.268   61%  71w | +0.273   61%  70w | +0.296   61%  71w | +0.261   61%  69w | +0.139   76%  34w
  10 | +0.000   64%   0w | +0.232   66%  76w | +0.378   68%  71w | +0.173   67%  72w | +0.348   68%  70w | +0.349   68%  70w | +0.372   68%  70w | +0.347   68%  70w | +0.344   68%  70w | +0.348   68%  64w
  30 | +0.000   64%   0w | +0.233   67%  77w | +0.299   65%  77w | +0.190   65%  77w | +0.303   65%  77w | +0.300   65%  77w | +0.300   65%  77w | +0.302   65%  77w | +0.300   65%  77w | +0.455   66%  76w
  60 | +0.000   65%   0w | +0.148   69%  78w | +0.253   65%  80w | +0.158   66%  79w | +0.259   65%  80w | +0.254   65%  80w | +0.255   65%  80w | +0.259   65%  80w | +0.257   65%  80w | +0.481   65%  79w

## pooled — mean Winkler skill vs floor (penalizes misses by 2/α: undercoverage is not free)
   n |    floor |   incumb |    best1 |     knn5 |     full |   puct16 |  exh@E16 |   puct64 |  exh@E64 |   served
   3 |   +0.000 |   -0.524 |   +0.000 |   +0.000 |   +0.000 |   +0.000 |   +0.000 |   +0.000 |   +0.000 |   +0.000
   5 |   +0.000 |   -0.526 |   -0.486 |   -0.465 |   -0.447 |   -0.471 |   -0.483 |   -0.447 |   -0.487 |   +0.030
  10 |   +0.000 |   +0.221 |   +0.407 |   +0.199 |   +0.379 |   +0.379 |   +0.401 |   +0.379 |   +0.374 |   +0.369
  30 |   +0.000 |   +0.267 |   +0.364 |   +0.245 |   +0.366 |   +0.365 |   +0.364 |   +0.366 |   +0.364 |   +0.445
  60 |   +0.000 |   +0.301 |   +0.381 |   +0.289 |   +0.393 |   +0.383 |   +0.384 |   +0.393 |   +0.389 |   +0.460
```

⛔ **Read Winkler, not interval-CRPS, at small n.** The uniform-on-interval CRPS charges a miss only 2× its distance, so an under-covering interval scores well: the incumbent leads the CRPS table at n=3/5 (+0.37/+0.42) while covering **40%**. Winkler (miss × 2/α) puts it at **−0.52** — below the floor.

Served-posture floor rate:

```text
served floor-posture rate per n:
  n=  3: 100.0%
  n=  5: 55.0%
  n= 10: 13.2%
  n= 30: 21.6%
  n= 60: 20.4%
```

## PUCT vs exhaustive at equal budget (n ≥ 10, all worlds)

```text
## PUCT vs exhaustive at EQUAL distinct-composition budget (n ≥ 10, all worlds)
  b= 16: mean E=14.1 | selection CRPS: puct better 129 / tie 1358 / worse 13 | held-out rel. gain of puct over exh -0.0122
  b= 64: mean E=33.5 | selection CRPS: puct better 118 / tie 1376 / worse 6 | held-out rel. gain of puct over exh -0.0031
  b=256: mean E=46.8 | selection CRPS: puct better 50 / tie 1448 / worse 2 | held-out rel. gain of puct over exh -0.0038
```

PUCT finds a lower SELECTION-split score more often than it loses (129/13 at b=16), and scores WORSE on held-out queries at every budget. It overfits the selection split harder. Per Issue 952's rule, **exhaustive ships and PUCT waits** for bet families too large to enumerate. Over a 7-bet roster at depth ≤ 3 (63 compositions), full enumeration costs 52 µs at n=60.

## G2 — cost (n = 60, loaded box)

```text
        knn5 fixed: calibrate     28.6 µs (E= 1, bets=1) | estimate p50    208 ns p99    292 ns
           puct@16: calibrate     44.2 µs (E=12, bets=7) | estimate p50     42 ns p99     84 ns
           puct@64: calibrate     49.2 µs (E=25, bets=7) | estimate p50     42 ns p99     84 ns
          puct@256: calibrate     60.2 µs (E=41, bets=7) | estimate p50     42 ns p99     84 ns
   full exhaustive: calibrate     52.3 µs (E=63, bets=7) | estimate p50     42 ns p99     84 ns
```

`estimate_into` p99 is 84 ns. p99 index 9 900 of 10 000 gives a tail support of 100. The search is a cold path (`calibrate`). The cached per-bet LOO columns make a composition score O(n·|mask|) additions, so full enumeration (52 µs) costs less than PUCT@256 (60 µs).

## G4 — zero-alloc served path

`bet_search::tests::estimate_into_is_zero_alloc_on_both_postures` passes in dev AND in `--release --features alloc_tracking`, on both the bet posture and the floor posture.

## Verdicts

1. **G1 floor-beat (split intact): PASS.** On Winkler the served posture is at or above the floor in every comps-size cell: +0.000 at n=3 (always demoted), +0.030 / +0.369 / +0.445 / +0.460 at n=5/10/30/60. It also beats the incumbent `EstimateHead` in every cell. The incumbent is below the floor at n=3/5 (−0.52/−0.53) and gets +0.22/+0.27/+0.30 at n≥10.
2. **Search-earns-its-budget: FAIL for PUCT → control (d) exhaustive ships.**
3. **Composition adds little over SELECTION.** best1 (singletons only) is within ±0.03 of full on Winkler at n≥10. The gain over the incumbent comes from choosing the right bet family. Composing bets adds almost nothing. The roster, not the tree, is the lever.
4. **The gate earns its keep in the shifted world.** In Drift every bet arm loses to the floor (−0.17 to −1.19 CRPS skill). The served posture demotes to the floor at n=30/60 (skill +0.000). Pooled served skill therefore rises above every raw arm at n≥30.
5. **The crossover is n = 5.** At n=5, raw bet arms are −0.45 to −0.49 on Winkler: undersized calibration (2–3 residuals) clamps the conformal index to the max residual. The gate rescues the cell to +0.03 by demoting 55% of tasks. At n=3 the split leaves fewer than 2 selection rows, so the floor serves by construction. The demote threshold is therefore DERIVED from this axis: the gate itself, not an `n < k` edge.

## Per-world tables

```text
## world Ratio
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   56%   0w | +0.305   49%  95w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w | +0.000   56%   0w
   5 | +0.000   74%   0w | +0.457   50% 100w | +0.823   75% 100w | +0.266   80%  93w | +0.798   76% 100w | +0.798   76% 100w | +0.823   75% 100w | +0.798   76% 100w | +0.823   75% 100w | +0.788   74%  95w
  10 | +0.000   77%   0w | +0.422   89%  99w | +0.840   83% 100w | +0.290   84%  92w | +0.835   84% 100w | +0.840   84% 100w | +0.840   83% 100w | +0.835   84% 100w | +0.840   83% 100w | +0.835   84% 100w
  30 | +0.000   80%   0w | +0.656   86% 100w | +0.865   81% 100w | +0.598   81% 100w | +0.865   81% 100w | +0.866   81% 100w | +0.865   81% 100w | +0.865   81% 100w | +0.865   81% 100w | +0.865   81% 100w
  60 | +0.000   80%   0w | +0.732   86% 100w | +0.876   81% 100w | +0.704   80% 100w | +0.877   81% 100w | +0.877   81% 100w | +0.876   81% 100w | +0.877   81% 100w | +0.876   81% 100w | +0.877   81% 100w

## world Plane
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   65%   0w | +0.408   52%  98w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w | +0.000   65%   0w
   5 | +0.000   76%   0w | +0.499   51% 100w | +0.078   74%  76w | +0.331   72%  93w | +0.159   75%  81w | +0.160   74%  83w | +0.074   75%  76w | +0.158   75%  80w | +0.072   75%  75w | +0.049   76%  47w
  10 | +0.000   80%   0w | +0.426   85%  98w | +0.941   85% 100w | +0.278   81%  89w | +0.941   85% 100w | +0.941   85% 100w | +0.941   85% 100w | +0.941   85% 100w | +0.941   85% 100w | +0.941   85% 100w
  30 | +0.000   73%   0w | +0.662   88% 100w | +0.954   82% 100w | +0.613   83% 100w | +0.954   82% 100w | +0.954   82% 100w | +0.954   82% 100w | +0.954   82% 100w | +0.954   82% 100w | +0.954   82% 100w
  60 | +0.000   79%   0w | +0.760   87% 100w | +0.956   80% 100w | +0.732   81% 100w | +0.956   80% 100w | +0.956   80% 100w | +0.956   80% 100w | +0.956   80% 100w | +0.956   80% 100w | +0.956   80% 100w

## world Smooth
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   66%   0w | +0.435   47%  99w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w | +0.000   66%   0w
   5 | +0.000   80%   0w | +0.470   50%  98w | +0.259   73%  82w | +0.372   73%  94w | +0.245   73%  78w | +0.147   73%  78w | +0.250   73%  80w | +0.244   73%  78w | +0.234   73%  78w | -0.011   77%  11w
  10 | +0.000   79%   0w | +0.310   75% 100w | +0.127   81%  77w | +0.293   81%  97w | +0.091   82%  72w | +0.105   81%  73w | +0.099   81%  72w | +0.088   82%  72w | +0.079   81%  72w | +0.064   80%  55w
  30 | +0.000   80%   0w | +0.289   84%  97w | +0.279   80%  95w | +0.315   79% 100w | +0.271   80%  93w | +0.267   80%  93w | +0.275   80%  95w | +0.271   80%  93w | +0.264   80%  93w | +0.253   80%  87w
  60 | +0.000   81%   0w | +0.366   88% 100w | +0.336   80%  99w | +0.350   80% 100w | +0.337   80%  99w | +0.335   80%  99w | +0.337   80% 100w | +0.337   80%  99w | +0.336   80%  99w | +0.334   80%  98w

## world Flat
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   67%   0w | +0.892   52% 100w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w | +0.000   67%   0w
   5 | +0.000   80%   0w | +0.890   52% 100w | +0.597   78%  94w | +0.864   76% 100w | +0.633   78%  95w | +0.579   77%  94w | +0.576   79%  94w | +0.633   78%  95w | +0.553   78%  94w | -0.024   78%  19w
  10 | +0.000   80%   0w | +0.203   78%  82w | +0.153   84%  77w | +0.185   84%  82w | +0.051   84%  76w | +0.031   84%  76w | +0.154   84%  78w | +0.051   84%  76w | +0.031   84%  76w | +0.029   84%  64w
  30 | +0.000   80%   0w | +0.193   78%  88w | +0.202   82%  91w | +0.177   82%  86w | +0.209   82%  93w | +0.208   82%  93w | +0.209   82%  91w | +0.209   82%  93w | +0.209   82%  93w | +0.204   82%  91w
  60 | +0.000   79%   0w | +0.156   80%  90w | +0.236   82% 100w | +0.192   82%  96w | +0.244   81% 100w | +0.242   81% 100w | +0.244   81% 100w | +0.244   81% 100w | +0.244   81% 100w | +0.241   80%  99w

## world Drift
   n |             floor |            incumb |             best1 |              knn5 |              full |            puct16 |           exh@E16 |            puct64 |           exh@E64 |            served
   3 | +0.000   78%   0w | -0.211    2%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w | +0.000   78%   0w
   5 | +0.000   79%   0w | -0.219    2%   0w | -0.355    5%   0w | -0.286    1%   0w | -0.354    5%   0w | -0.347    6%   0w | -0.359    5%   0w | -0.354    5%   0w | -0.375    6%   0w | -0.106   72%   0w
  10 | +0.000    5%   0w | -0.200    3%   0w | -0.171    5%   0w | -0.182    4%   0w | -0.177    5%   0w | -0.172    5%   0w | -0.174    5%   0w | -0.177    5%   0w | -0.173    5%   0w | -0.127    5%   0w
  30 | +0.000    5%   0w | -0.637    1%   0w | -0.804    0%   0w | -0.753    1%   0w | -0.787    1%   0w | -0.797    1%   0w | -0.803    0%   0w | -0.792    1%   0w | -0.794    1%   0w | +0.000    5%   0w
  60 | +0.000    4%   0w | -1.274    6%   0w | -1.140    2%   0w | -1.186    4%   0w | -1.117    4%   0w | -1.141    2%   0w | -1.140    2%   0w | -1.118    4%   0w | -1.129    3%   0w | +0.000    4%   0w
```

## Follow-ups (not done here)

- The consumer swap (`EstimateHead` → `BetSearchHead` as the F0 estimate path) is Plan 628 T0.4 territory, under the `katgpt-rs-fathom-phase0` lease. This bench is the evidence for that call; this session did not make it.
- Issue 952 T5 (wasm32 attestation packs) stays owner-gated.
