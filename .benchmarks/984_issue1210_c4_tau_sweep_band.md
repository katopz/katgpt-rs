# Bench 984 — Issue 1210 C4: the escalation-band τ sweep (post-hoc on the frozen 983 field)

**Status:** RECORD — the C4 instrument EXECUTED; REPORT gate (monotone decay asserted as well-formedness, knee disclosed, band table delivered). Measured 2026-11-15, amo (Mac17,15 / M5 Ultra, AC power), `CARGO_TARGET_DIR=/tmp/i1210c4` isolated; training/capture figures byte-identical across the two runs (λ 0.311, 51,284 rows / 46,804 armed / 1,620 spikes — the Bench-983 field exactly; the inter-run delta was the grid-precision fix below).

**Rig:** `../riir-ai/crates/riir-poc/benches/escalation_band_tau_sweep.rs` — `cargo bench -p riir-poc --bench escalation_band_tau_sweep` (~4 min incl. training; SEED 1210 / TRAIN_SEED 42, the 983 posture byte-for-byte — the capture IS the 983 field, rebuilt deterministically in-memory; disclosed, no persisted artifact). λ_ref = **0.311** identical.

**Instrument:** τ → escalation-rate curve over a 200-point grid on (0, ln2] plus τ=0, rate(τ) = P(drift > τ) over armed rows; knee = the grid point of maximal log-curve curvature; band table = the τ achieving each target escalation rate (the ESC consumer's dial, the decision-level analog of `seat_rerank`/ESC escalation-band tuning). **The frozen-field premise holds:** the entire sweep is post-hoc — one capture, every τ free.

## En-route: the grid-precision fix (a wrong-table class, caught by the first run's own output)

The first run's grid used `std::f64::consts::LN_2` — one ulp SMALLER than the f32 `LN_2` the kernel stores — so the exactly-ln2 spike mass (~10% of spikes) read as still "exceeding" the top grid point: a phantom 10.4% rate floor that made the spike-only band table report 10%/5%/2%/1%/0.5% all "unreachable". Fixed (grid on `f32::LN_2 as f64`, the drift values' own scale), re-run: rate(top τ) = 0.00e0 everywhere and the spike table resolves. Recorded because it is the same defend-wrong class as 983's label swap: correct numbers feeding a wrong CONSUMER table.

## Measured

Field shape: 46,804 armed rows, spike mass 1,620 (**3.46%**); spike quantiles p50 0.2311 · p90/p99/max **0.6931 = ln 2 exactly** — the spike mass is bimodal (≈half at 0.23, most of the rest pinned AT ln2 = disjoint-set flips), which is why the knee sits between the modes.

| read | rate(τ=0) | knee | band table (τ → escalation rate) |
|---|---|---|---|
| **POOLED** (46,804) | 0.0346 | τ ≈ 0.4817 (rate 0.59%) | 2%→0.2022 · 1%→0.4041 · 0.5%→0.5095 |
| **SPIKE-ONLY** (1,620) | 1.0000 | τ ≈ 0.4817 (rate 16.98%) | 25%→0.4367 · 10%→0.6898 · 5%→0.6915 · 2%→0.6925 · 1%→0.6928 · 0.5%→0.6930 |
| Alt (10,181) | 0.0429 | τ ≈ 0.4817 | 2%→0.2429 · 1%→0.4581 · 0.5%→0.5062 |
| PairSwap (9,044) | 0.0376 | τ ≈ 0.5580 | 2%→0.2288 · 1%→0.4512 · 0.5%→0.5226 |
| Fib (9,554) | 0.0287 | τ ≈ 0.3951 | 2%→0.1550 · 1%→0.3474 · 0.5%→0.4995 |
| Cycle4 (8,252) | 0.0325 | τ ≈ 0.5476 | 2%→0.1905 · 1%→0.3773 · 0.5%→0.5367 |
| Noise (9,773) | 0.0308 | τ ≈ 0.5303 | 2%→0.1704 · 1%→0.3780 · 0.5%→0.4906 |

Rates ≥ 10% are **unreachable at any τ ≥ 0** in the pooled read — the field's own ceiling is rate(τ=0) = 3.46% (the spike mass); the table prints τ 0.0000 there (the honest answer: τ=0 is the max-escalation posture).

## Verdict: REPORT — the deliverable is the dial, not a pass

- The monotone-decay gate held by construction (strict-`>` exceedance over a fixed multiset is non-increasing — asserted in-sweep, never fired). The issue's gate wording is satisfied as a well-formedness proof; nothing about the field is "proven" by it.
- **The ESC-consumer read:** at this posture the τ dial is usable only in the 0.5–3.5% escalation band (pooled); the knee (τ ≈ 0.48, between the spike modes at 0.23 and ln2) is where the curve turns. Below-knee τ spends escalation on the 0.23-mode spikes; above-knee τ buys precision at <1% rates but is ulp-close to the disjoint-set ceiling — the band table names exact τs per target rate.
- **Family contrast (disclosed):** Alt (the learned family) carries the HIGHEST τ=0 rate (4.29% — the most drift events, consistent with 982's "learned family flips more") but a mid knee; Fib's knee is the LOWEST (0.3951) — its spikes arrive at lower τ. No promotion claim: Bench 802's G3 fences the joint-gate family at micro scale, and the ESC arming is owner-gated.
- **The stationary-field caveat is the instrument's own premise:** a real two-lane serving posture must re-freeze its capture (the field under distribution drift moves; this capture is a micro fixture).

## What C4 leaves open

- **C5** stays trigger-gated (the plan-623 cross-host activation channel).
- **C7** (verify-cache posture) staged-unclaimed.
- Consumer wiring (ESC/seat_rerank reading the band table) is the owner-gated step the issue itself names.
