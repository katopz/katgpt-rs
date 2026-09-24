# Bench 884 — exact_mass_admit GOAT gate: cost-vs-calibration vs the hard-cut family (Issue 879 T3 / Research 584)

**Status:** COMPLETE — G1 PASS · G2 PASS (record + calibration assert + gross ceiling) · G3 clean · G4 PASS. Feature `exact_mass_admit` (katgpt-core + root forward) stays **OPT-IN** — no default promotion (the T4 posture: the calibration gain is real and measured, the consumer wiring is the missing half, and none was assumed).

**Box:** 4090RTX workstation (Windows 11, i7-13700K), AC power, no concurrent compute observed at run time (CPU 12%). Release profile. Timing via the shared `best_of_arms` harness (round-robin interleave + per-arm MIN — the load-invariant form, Issue 833's treatment).

## What was measured

The MAttr "sigmoid top-k" operator (`exact_mass_admit_into`: bisect τ until Σσ((s−τ)/T) = k, emit the soft mask) against the hard-cut baseline family, at the issue's N ∈ {1e3, 1e5, 1e7}, k = N/10, T = 1, uniform scores in [−3, 3):

- **gate** — `katgpt_spectral::manifold_power_iter_router::gate_sigmoid_topk_into` (per-expert sigmoids + selection-sort hard cut). Run ONLY at N=1e3 — its own doc bounds the O(k·N) selection sort to game-scale N; at 1e5 it is 1e9 comparisons/call, absence-of-regime rather than a measurement.
- **hardcut** — `select_nth_unstable` top-k + `exact_sigmoid_f64` weights on the winners (the generic arbitrary-k hard cut).

## Results

| N | arm | best µs/call | \|Σm − k\| (calibration) |
|---|---|---|---|
| 1e3 | **ours** | **69.8** | **5.16e-7** |
| 1e3 | gate | 77.1 | 6.17 |
| 1e3 | hardcut | 1.5 | 6.17 |
| 1e5 | **ours** | 10,357 | **4.08e-5** |
| 1e5 | hardcut | 229 | 637.6 |
| 1e7 | **ours** | 1,095,014 | **6.98e-3** |
| 1e7 | hardcut | 25,031 | 63,736 |

Findings:

1. **Calibration is the mechanism and it holds at scale**: ours pins \|Σm−k\| to ~1e-8 RELATIVE of k at every N (5.16e-7 at k=100 → 6.98e-3 at k=1e6). Both baselines drift −6.4% below k at every N — the uncalibrated sigmoid weights sum to ≈0.936·k on this score distribution, and that drift is distribution-dependent (a different score family moves it). The hard cut admits the right COUNT; the weights it emits are not a budget.
2. **At the router's designed regime the calibrated operator is CHEAPER than the shipped gate path**: 69.8 vs 77.1 µs at N=1e3 (0.906×) — the bisection's ~32-50 scalar passes cost less than the gate's O(k·N) selection sort at k=100. In-regime, exact mass is not a premium feature.
3. **Against the generic hard cut the cost is real**: ~45× at 1e5/1e7 (10.4 ms vs 0.23 ms; 1.10 s vs 25 ms) — ~2.2 ns per expit·element across the bisection passes. This is the offline/calibration-tier posture of the issue: a `DensityBudget` ladder boundary or a `thermal_lod` elbow sweep runs once per calibration cycle, not per token.
4. **Gross regression ceiling**: ours @1e7 must stay under 2.5 s (~2.5× the measured 1.10 s) — catches de-optimization, not box noise (the bench-845 pattern).

## Gates

- **G1** — sum-to-k at every N (tol 1e-3 + 2e-7·N) + masks in [0,1]; the heavy suite (shift-invariance bit-identity, nestedness-in-k, extremes, degenerate scores, tiny-T, LogFrontier protocol) lives in `crates/katgpt-core/tests/exact_mass_admit_g1.rs` (14/14 PASS).
- **G2** — the table above; calibration pinned at every N (hard assert); relative latency RECORDED, never asserted (the no-assumed-win rule); gross ceiling (hard assert).
- **G3** — feature-off: default clippy unchanged (the modules cfg-gate to nothing; wasm32 `--features exact_mass_admit --lib` compiles clean — checked on this box).
- **G4** — `exact_mass_admit_g4_alloc`: 500 × N=4096 operator calls + 100k LogFrontier sample/observe → **0 allocations** (PASS).

## Verdict (Issue 879 T4)

- **No consumer gain was measured** (no consumer was wired — that was the POC's scope line), so per the no-assumed-win rule the feature stays opt-in and the issue closes with the recorded tradeoff: **exact mass at router-regime cost (finding 2), machine-precision mass control (finding 1), 45× vs the generic hard cut at bulk N (finding 3)**.
- Consumer postures for any future promotion (each needs its own gate; the calibrated-mass upgrade of `gate_sigmoid_topk` is the cheapest — finding 2 says the cost is already competitive in its regime): (a) `gate_sigmoid_topk` calibrated-mass upgrade (katgpt-spectral router gates); (b) `block_topk` calibrated gate mass; (c) `cs_kv_probe::GatedKvSlice` exact-k log-bias sweep; (d) riir-ai `memory_soup` SSC top-k gates.
- Explicit non-lane: no prefill/decode hot-path claim (attribution is quality-side; the 4090 prefill league is kernel-side).

## Reproduce

```bash
cargo test --features exact_mass_admit --test bench_884_exact_mass_admit_goat --release -- --nocapture
cargo test -p katgpt-core --features exact_mass_admit --test exact_mass_admit_g1
cargo test -p katgpt-core --features exact_mass_admit --test exact_mass_admit_g4_alloc --release
```

Cross-refs: [Research 584](../.research/584_Matryoshka_Attribution_Sigmoid_Topk_Budget_Primitives.md) · resolution record: [HISTORY.md §Issue 879](../HISTORY.md) (issue file removed per the noise-reduction rule) · upstream arXiv:2609.25518 (math only; the code repo carries no license).
