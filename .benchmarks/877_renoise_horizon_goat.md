# 877 — renoise horizon-weighting GOAT (Issue 875 T2)

**Status:** GOAT G1 / G2-QUALITY / G2-LATENCY / G3 / G4 ALL PASS (2026-09-22) — lands OPT-IN as a combined-gate surface (`renoise_ce` + `horizon_weights`; `renoise_ce` itself is katgpt-core default-on and its incumbent path is byte-untouched). NOT PROMOTED: T5 owns the promotion verdict; `horizon_weights` stays default-off, and the mode's `tau` does not transfer from the incumbent's calibration (documented in-module).

Owner: `Issue 875` T2 · Source: [Research 582](../.research/582_Probability_Flow_Distillation_Wasserstein_Gradient_Flow.md) (arXiv:2605.09071 Probability-Flow Distillation, Theorem 1's (T−t) Fubini factor) · T1 substrate: commit `02d5813d` (the T1 weights/table substrate — no bench doc of its own; this bench covers the CONSUMER).

## Box state (the G2 rule)

- Windows 11 Pro, i7-13700K (16 cores), RTX 4090 idle, AC power.
- CPU 7% at session start; ≥2 sibling agent sessions active throughout (riir-clippy Batch-172 corpus work in another repo + a katgpt-rs doc/reconcile lane in the MAIN checkout — this gate ran in an isolated `git worktree` at `.raw/k875t2` (detached @ `876b61f8` + this change), own `CARGO_TARGET_DIR=C:/tmp/k875t2`).
- Release profile for the latency gate; debug for correctness arms (deterministic, identical numbers both profiles).

## What landed

`renoise_ce_score_horizon` + `RenoiseCeHorizon` (combined-gate: `#[cfg(feature = "horizon_weights")]` inside the default-on `renoise_ce` module) — the k-draw budget reallocated over `[0.02·L, 0.98·L]` (L = `perturbation_level`, the horizon) sampled inverse-CDF from the remaining-horizon density `∝ (L − t)` via `horizon_weights::remaining_horizon_t_sample` (new T1 substrate: exact inverse CDF, one `sqrt`, zero alloc). Averaging is flat over the tilted draws — the weights ride the sampling distribution (importance sampling), never double-applied as explicit per-draw weights. Same NFE budget, same loop shape, same `[f32; 8]` per-draw record.

## G2-QUALITY — planted-drift selectivity oracle (the promotion-deciding axis)

Synthetic mixture, deterministic fastrand seed 87502: 32 stable candidates (basin-resident, jitter σ≈0.008) + 16 easy planted (offset ∈ [0.15, 0.30]) + 16 marginal planted (offset ∈ [0.04, 0.09]) over a two-center basin probe ({0, 10}, α=0.9 partial snap, house sum-of-3-uniforms perturbation ×2.0). Boundary crossing is ~6σ at all sampled levels — the differentiator is the FINE-GRAINED floor mechanism, not outlier robustness. precision@32 (fraction of the 32 lowest-drift-scored that are truly stable), three arms at k=8 draws each:

| arm | precision@32 |
|---|---|
| incumbent `renoise_ce_score` @ fixed t=0.40 | **0.906** |
| uniform-range control (same range, flat sampling) | **0.938** |
| `renoise_ce_score_horizon` (the law) | **1.000** |

Bars (all pass): horizon ≥ 0.90 absolute; horizon ≥ incumbent + 2 candidates (measured +3); horizon ≥ uniform control (+2). The uniform-range control isolates the LAW from the RANGE — tilted sampling beats flat sampling over the SAME range, so the gain is the (T−t) weighting itself.

Mechanism (why the law wins this class): a fine-grained planted offset δ carries a t-INDEPENDENT drift signal (~0.81·δ² through the partial-snap probe) while the noise floor of every measurement grows ∝ t². The tilted density's effective t² is ≈0.173·L² vs the incumbent's L² — the floor collapses ≈5.8× at the same budget, and both classes' score variance shrink with it. The incumbent spends its entire k-draw budget at the single noisiest admissible point.

Parameter provenance: swept in a python mirror of this exact probe (`m ∈ {1,2,3,4} × 4 marginal bands × 60 seeds`); the chosen band (m=2, marginal [0.04, 0.09]) had horizon ≥ fixed + 2 candidates in **60/60** seeds with zero losses; ordering horizon > uniform > fixed held in every configuration measured. The Rust gate's single fixed seed sits mid-distribution (its fixed-arm 0.906 vs the sim's 60-seed mean 0.834 — comfortably inside the band).

## G2-LATENCY — unchanged class

Shared interleaved-pairs `ab_timing` harness (Issue 723/833 treatment), 41 rounds × 512 iters/arm, D=8 states, k=8, RELEASE profile:

- incumbent 535.96 ns/score vs horizon 533.06 ns/score — **median ratio 0.9916 (−0.8%)**, rounds 0.956–1.106, 41/41 survived. Bar ≤ 1.25. The sampler's one `sqrt` + FMAs per draw is invisible under the clone + perturb + re-resolve + drift body at equal budget.

## G1 / G3 / G4

- **G1**: same-seed determinism bit-pinned; sampler endpoints/monotonicity/CDF-bucket (8192 draws, 10 equal-CDF buckets, ±20%)/f64-oracle pinned in `horizon_weights::tests`; horizon-mode level-range + fallback behavior pinned in `renoise_ce::tests::horizon` (11 new in-module tests).
- **G3**: default path untouched — `renoise_ce_score` body unchanged; the mode compiles to nothing when `horizon_weights` is off. Measured: default katgpt-core lib suite **2063/0**; `bench_406_renoise_ce_goat` **5/5**; `cargo check -p katgpt-core --no-default-features` clean; clippy `-D warnings` clean at default, at `--features horizon_weights`, and on the new root test target.
- **G4**: zero allocation on the horizon score path with a fixed-array State (asserted in-module under `debug_assertions`/`alloc_tracking`); same per-draw clone story as the incumbent.

## Honest caveats

- The oracle regime is the paper's MOTIVATING defect class (fine-grained detail that survives only at low noise — why PFD weights (T−t)). Defect classes visible ONLY at high t (e.g. mask-perturbation repair of wrong-but-confident tokens) favor a high fixed level; the mode is opt-in and the doc names the regime. Basin-crossing regimes benefit FURTHER (the law's w(T)=0 never spends budget in the crossing band).
- A structural near-coincidence worth knowing: the (T−t)-tilted RMS of t over [0, T] is ≈0.408·T — the paper's fixed t=0.40 IS approximately the tilted-RMS operating point, which is why a fixed 0.40 is a strong single-point surrogate. The win here comes from anchoring the range at the incumbent's own level L=0.40 as the horizon (the mode samples strictly BELOW it, cap 0.98·L).
- `tau` calibrated for the incumbent does NOT transfer (the horizon mean sits lower — smaller floors at low noise). Calibrate per mode.
- Synthetic probe, single deterministic seed in-gate (60-seed sim evidence behind it); not a production-distribution claim. T5 weighs promotion on this + any production consumer evidence.
