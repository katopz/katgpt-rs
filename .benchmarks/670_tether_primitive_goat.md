# Bench 670: `tether` primitive GOAT — closed-form outcome-fit estimator blend

**Date:** 2026-08-21
**Issue:** `katgpt-rs 675` — resolved + removed in the same commit; this doc is the record.
**Source:** [Research 426](../../riir-train/.research/426_Le_Critique_PVF_TETHER.md) — arXiv:2608.16739 "Le Critique" (TETHER baseline).
**Surface:** `katgpt_core::tether` (opt-in feature `tether`) — `TetherBlend` (ρ\* OLS + EMA + lag-law API shape), `fit_rho`/`sse` (batch forms), `EvAccumulator` (one-pass Welford EV), `control_variate_improves` (the iff gate, batch), `horizon_decay` (λ = c^(1/L) LUT).

```bash
cargo test  -p katgpt-core --features tether --lib tether::       # 15 fixtures
cargo clippy -p katgpt-core --features tether --lib               # 0 warnings
cargo clippy -p katgpt-core --features tether --bench tether_bench # 0 warnings
cargo bench  -p katgpt-core --features tether --bench tether_bench -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 30
cargo test  -p katgpt-core --lib                                   # default G3
```

## VERDICT: GOAT G1–G4 ALL PASS — ships opt-in; promotion = the Plan 345 consumer GOAT

Semantics were validated by a real consumer **before** landing (riir-clippy
Bench 042, commit `5494dbe` — the inlined copy of exactly T1/T2's math ran
the full A/B): the in-sample guarantee held exactly on real recorded streams,
ρ drift was large + reproducible, the fit was heap-free, determinism was
bit-exact at the pipeline seam, and the API shape prevented same-window
application with no debug-assert. That consumer's **metric** measured
negative — recorded in-source as the prediction-vs-ranking hazard (see
below), which is exactly why the primitive stays opt-in and why a ranking
consumer must never cite it.

| Gate | Result | Detail |
|---|---|---|
| G1 fixtures + determinism | **PASS** | 15/15 first run: grid-argmin equivalence, exact never-worse both endpoints, complementary/identical/anti-complementary regimes, ρ-frozen-in-window (lag law), known-answer EMA (bit-identical replay + geometric bound), degenerate hold, EV one-pass vs two-pass (1e-12 rel), control-variate iff (informative=true / noise=false / degenerate=None), λ round-trip 1e-9, holdout, drift tracking, admissibility MC pair |
| G2 latency (M3 Max, release) | **PASS, ~45× headroom** | observe K=16 **2.58 ns**, K=64 **2.22 ns** (close+EMA amortized inside), blend **1.06 ns**, horizon_decay ≈ 5.1 ns — gate ≤ 100 ns/observe at K ≤ 64 |
| G3 no-regression | **PASS** | default lib **1904/0/7** unchanged; default `cargo check` clean; clippy 0 with the feature (lib + bench targets) |
| G4 allocation | **PASS** | 0 allocs / 0 bytes over 10 000 observes + closes + EMA publishes + telemetry + EV reads + decay lookups (per-thread `TrackingAllocator`) |

### Fixture decisions worth recording

- **The holdout margin was retuned before landing.** The first construction
  (r = 0.6u + 0.2n, p2 = 0.5u + 0.3n) puts the population-optimal ρ at 0.88 —
  so close to the p2 endpoint that the blend beats p2-only by only ~3.4%,
  inside MC noise once the EMA lag penalty is paid. Retuned to
  r = 0.6u + 0.1n, p2 = 0.5u + 0.5n → ρ_opt ≈ 0.6, blend beats p2 by ~30%,
  p1 by ~49%. A holdout fixture whose margin the noise can flip is not a
  fixture.
- **`assert_eq!(fit_rho, 0.5)` on the complementary regime is f32-fragile**
  (r − (r−0.1) is not exactly 0.1 in f32); pinned with 1e-6 tolerance
  instead. The clip-to-1.0 (anti-complementary) and degenerate-hold asserts
  are exact and stay exact.
- **DEFAULT_RHO = 0.0** (not the consumer's legacy 0.4): no evidence → do
  not blend. riir-clippy pins its historical 0.4 via `with_params`; Plan 345
  pins ρ₀ = 0 with EV-gated movement — both documented on the constant.

## Task dispositions (Issue 675)

- **T1–T5 [x]** — all fixtures listed in the issue ship in-module
  (`tether::tests`), including the two the issue sourced from Bench 042's
  validated list (never-worse on real streams, ρ-frozen-in-window).
- **T6 [x]** — G1–G4 above; both hazard comments in-source (module docs:
  HAZARD 1 Report-the-Floor, HAZARD 2 prediction-vs-ranking with the Bench
  042 numbers).
- **T7 [x]** — ruling: **stays opt-in** (the CLR precedent — promotion
  requires a production consumer's own GOAT; Plan 345 Phase 2/3 is the
  candidate). Recorded in Research 426 (riir-train) in the same session.

## Consumers

1. **riir-train `loss_grpo` TETHER baseline (Plan 345)** — unblocked by this
   landing; Plan 345 T1.1's sanctioned inline (30-line ρ\* copy) is now
   deletable in favour of `katgpt_core::tether`.
2. **riir-clippy selection blend** — measured NEGATIVE (Bench 042), kept as
   the reproducible artifact + the in-source hazard. Its local inline
   (`src/selection_tether.rs`) is delete-on-landing by design; the swap to a
   katgpt-core re-export shim is **deferred behind that repo's active
   sibling WIP** (Cargo.toml is mid-flight in another session — the swap
   needs a feature-flag dep edit and must not land through it).

## Numbering note

`.benchmarks/.highwater` was found stale at 666 with 667–669 present (the
exact dual-allocation trap the numbering discipline warns about); re-scanned
at write time, this doc takes **670**, and the highwater is corrected in the
same commit.
