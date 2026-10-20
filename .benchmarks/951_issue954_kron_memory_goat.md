# Bench 951 — `kron_memory` GOAT gate (Issue 954)

**Date:** 2026-10-20 · **Box:** M3 Max (16 cores), macOS 26.6.2, AC power (battery 100%), ~28% CPU load at launch (sibling agents active) · `--release`, isolated `CARGO_TARGET_DIR` · deterministic LCG seeds, re-run byte-identical.

**Run:** `cargo bench -p katgpt-core --features kron_memory,kron_chunk_wy,conformal_predictive_intervals --bench bench_951_kron_memory_goat -- --nocapture`

## Verdict: G0 FAIL — the falsifier fired; the flag stays opt-in forever

The headline claim (surgical unlearning beats baselines on distributed keys) is **refuted at the tested operating point**: at matched state bytes (16 KB) and 48 stored pairs (1.2% slot load), the plain one-hot slot table wins BOTH G0 statistics. This is the issue's pre-registered valid outcome ("failure on either statistic against either baseline keeps the flag opt-in forever"). The cell ships as a diagnostic-grade opt-in primitive: every G1 property test passes (the machinery is sound), but the promotion claim does not survive its falsifier.

## G0 — the falsifier (paired collateral-per-erase, distributed keys)

400 seeded trials × 3 arms at matched 16 KB state: Kron-erase (`unlearn_audited`, Newton step), vector-delta-erase-with-audit (`delta_mem` rank 64 at the matched Newton step — β=1 IS the vector-target Newton step for normalized keys), one-hot slot table (argmax addressing).

| arm | mean \|Δ\|/erase | post-erase recall @ε=0.05 |
|---|---|---|
| **kron** | 0.00482 | 0.780 |
| delta-mem (vector Newton erase) | 0.08715 | 0.008 |
| **one-hot table** | **0.00016** | **0.994** |

Paired (baseline − kron): delta-mem **+0.08233, LB95 +0.08141 > 0 ✓** — the double-gated erase IS 17× more surgical than the vector-target Newton erase on distributed keys (the interference-kernel story holds); one-hot **−0.00467, LB95 −0.00494 < 0 ✗** — the table wins.

**Reading (honest, with the capacity-trade lens):** the arms sit at their contracts' natural loads — Kron stores 48 *scalars* in 4096 slots (1.2%), the vector arm 48 *vectors* = 3072 scalars (75%), the table 48 of 4096 slots (1.2%) with argmax-lossy addressing. At 1.2% load a slot table is nearly lossless (0.994 recall; collisions ≈ P²/2/4096 ≈ 0.28/trial) and its erase is a slot zeroing. Kron's protective regime (dense superposition, high load) was not the regime G0 specified, and the pre-registered statistic does not bend post-hoc. **The measured complementarity:** kron beats the VECTOR contract 18× on collateral at 18× the recall (0.780 vs 0.008 — the vector contract is crushed at its 75% load), but loses to the TABLE at low load. Re-open bar (recorded, not run): a G0′ at matched load ≥ 50% with its own pre-registration — different claim, owner-gated.

## G2 — latency (--release, black_box args + receiver)

| gate | measured | bar | verdict |
|---|---|---|---|
| `write_newton` @64×64 | **1017 ns** | ≤ 2 µs | PASS |
| `read` @64×64 | **1010 ns** | ≤ 2 µs | PASS |
| `collateral_audit` @64-pair registry | **2867 ns** | ≤ 10 µs | PASS |
| chunk speedup @L=64 | **0.56×** (seq 90.1 µs vs chunk 161.7 µs) | ≥ 2× | **FAIL** |

The chunk lane is SLOWER than sequential: the fused apply + Gram triangles are ~1.5× the MACs of the sequential row axpys, and the sequential path's per-row contiguity already vectorizes well — the projected latency win does not exist on this hardware at this shape. `kron_chunk_wy` stays opt-in, UNPROMOTED; its equality-vs-sequential tests pin the tolerance contract for any future GEMM-shaped rewrite (re-open bar: blocked-GEMM Grams via the `kron_tile`/`simd_matmul` family — priced, not attempted).

## G4 — allocation

Zero allocations across `read` / `write_newton` / `collateral_audit` / `undo_last` / `write_snap` (counting allocator + the Issue-714 liveness canary). **PASS.**

## G5a — analytic interference envelope vs measured erosion

The self-consistent first-order envelope `MSE ≈ σ²_c/(1 − x)`, `x = P·β²/(d_k·d_v)` (each subsequent Newton write adds `E[r²]/(d_k·d_v)` to a stored pair's error; `E[r²] ≈ σ²_c + MSE`):

| P | measured MSE | analytic | ratio | band [0.5, 2] |
|---|---|---|---|---|
| 256 (x=0.0625) | 0.2954 | 0.3556 | 0.83 | IN |
| 1024 (x=0.25) | 0.2966 | 0.4444 | 0.67 | IN |
| 4096 (x=1) | 0.6022 | ∞ (model limit) | — | declared limit |

**2/3 in band → the analytic sizer is PROMOTED for the sub-saturation regime** (`P·β² < 0.98·d_k·d_v`). Measured mechanism worth recording: the erosion SATURATES at ≈ σ²_c (a stored pair's read decorrelates from its target as the state superposes — the "capacity" of a no-decay scalar cell on random addresses is the interference floor, not a retention cliff). The first envelope draft (`σ²·x/(1−x)`, missing the base-variance term) measured 13× off — the derivation error was fixed BEFORE recording; the verdict above is the corrected model's.

## G5b — Report the Floor

The residual-signal CI candidate (read ± q·residual_ema, q calibrated at 90% on the cal slice) vs the canonical floor `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (m=1, HStep, no decay), both observing every tick (the floor was given its full calibration slice — the fair posture):

| | coverage | Winkler ↓ | CRPS ↓ |
|---|---|---|---|
| candidate | **0.896** (nominal 0.90) | **2.624** | 2.156 |
| floor | 0.203 | 11.640 | **2.036** |

The candidate beats the floor on coverage (+0.69) and Winkler (4.4×) but loses CRPS (2.156 vs 2.036). The rule requires the floor beaten on coverage/Winkler/CRPS — **the CI is DROPPED; the cell ships signal-only** (`residual_signal()`), the pre-registered acceptable outcome. (The floor's 0.203 coverage is itself a finding: seasonal-naive is the wrong point forecaster on fresh-address associative streams — recorded for any future floor-harness consumer.)

## G1 / G3 (in-suite evidence)

- **G1:** 19 property tests, all passing (`cargo test -p katgpt-core --features kron_memory,kron_chunk_wy --lib kron_memory`) — Newton one-step hit, contraction sweep + the ≥2 divergence refusal, audit-kernel == measured collateral, orthogonal-either-axis zero interference, the frozen-state identity-fixed-point hash pin (64 replayed writes bit-identical), superposition linearity, single-level undo within `EPS_UNDO` + bit-exact snap undo + the loud capacity refusal, LS-snap steady state, correlated/rank-deficient health arms (gauges finite, measured independence violation 0.73 on a v=k stream), the norm-budget envelope fired at P > 4/λ̂ deterministically, registry ring mapping, zero-address refusals, residual-signal convergence.
- **G3:** default-features suite 2157 passed / 0 failed — byte-identical to the standing floor; the flag-off build compiles the module to nothing; no shipped module was modified (the only lib.rs/Cargo.toml edits are the feature rows + the gated `pub mod` + the linalg cfg-any membership, per the joined-at-birth rule).

## Session

katgpt-rs-kron-m3, 2026-10-20. Records: `.docs/03_memory/kron_memory.md` (the sizing + envelope laws), Issue 954 (closed+removed; HISTORY.md carries the record).
