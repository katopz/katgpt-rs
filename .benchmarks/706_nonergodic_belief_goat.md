# Bench 706: Nonergodic Belief Kernel GOAT Gate (`nonergodic_belief`)

**Status:** COMPLETE — G1–G4 ALL PASS (primitive-level); T2.5 recorded (beats both baselines on log-loss); stays OPT-IN per T3.2
**Date:** 2026-09-11
**Plan:** [`.plans/592_nonergodic_belief_kernel.md`](../.plans/592_nonergodic_belief_kernel.md)
**Research:** [`.research/545_Nonergodic_Belief_Decomposition.md`](../.research/545_Nonergodic_Belief_Decomposition.md)
**Source:** "The geometry of nonergodic composition" — Simplex blog, simplex.pub/nonergodic-geometry/ (2026-09-09); Mess3 matrices taken from its appendix A.2 (β=(1−α)/2, y=1−2x)
**Crate:** `katgpt-micro-belief` · feature `nonergodic_belief` (opt-in, NOT default)
**Machine:** M3 Max (macOS, aarch64), CPU-only. **No GPU gate needed** — pure f32 fixed-array modelless math; GPU-exclusivity rule N/A. Built/run in an isolated `CARGO_TARGET_DIR=/tmp/plan592` (shared `target/` contended by a sibling agent session).

---

## What shipped

`crates/katgpt-micro-belief/src/nonergodic.rs` (~880 lines): `ComponentModel`
trait (object-safe, Send+Sync, zero-alloc contract), `BernoulliCoin` +
`bernoulli_pair` (the blog's two coins, D=1), `Mess3Block` (blog appendix
A.2 closed forms, D=3, x ∈ (0, 0.5] so the blog's x=0.5 instance is
included), `SyntheticBlock` (bench-grade D≤64 deterministic HMM block),
`NonergodicFilter<const K, const D>` (η, log_w + derived linear w; per-tick
LSE renormalization), accessors `weights/component/telescope_into/
committed/revive_margin/revive_gate`. Zero-alloc hot path, `#[inline]` hot
fns, fixed-order deterministic LSE.

## GOAT gate verdict

| Gate | Bar | Measured | Verdict |
|---|---|---|---|
| **G1 exactness** | filter posterior == brute-force definition, 1e-5, L≤12 | exact match on ALL of: 2 coins blog (p=0.5/0.7) exhaustive L=12 (4096 seqs); 3 coins exhaustive L=12; 3 coins non-uniform prior exhaustive L=10; blog Mess3 pair (0.6,0.15)+(0.66,0.5) exhaustive L=8 (6561); 3-Mess3 exhaustive L=7 (2187); 5-Mess3 exhaustive L=6 (729); + 2500 seeded random L=12 Mess3 sequences across K∈{2,3,5} | **PASS** |
| **G1 identities** | Σw=1 per tick; telescope block sums == w_n; collapse monotone in expectation; revival | Σw and telescope asserted on every enumerated sequence; mean max w @10/30/60 = 0.676/0.908/0.973 strictly increasing (400 streams); revival: w flips 0→1 across a contradiction stream, exceeding the collapse floor (≈1e-25) | **PASS** |
| **G2 latency** | tick < 1000 ns @ K=8/D=8 | **564.1 ns** (1.8× headroom) | **PASS** |
| **G3 no-regression** | default-feature surface unchanged | default-features lib tests: **54 passed** (identical to pre-change count — the new module is cfg'd out); with feature: **68** (+14 nonergodic unit tests, +0 changes to existing); existing bom_sampling paths untouched | **PASS** |
| **G4 alloc-free** | 0 allocs after setup | **0 allocations** at all 9 grid points over 20 000 ticks each (CountingAllocator in the bench binary) + dedicated G4 test: construction/ticks/telescope/readouts all 0 | **PASS** |

G2/G4 grid (bench profile, warm 2 000 + timed 20 000 ticks, SyntheticBlock models):

| K | D | ns/tick | allocs |
|---:|---:|---:|---:|
| 2 | 8 | 161.4 | 0 |
| 2 | 32 | 631.6 | 0 |
| 2 | 64 | 1358.4 | 0 |
| 8 | 8 | **564.1** | 0 |
| 8 | 32 | 2600.4 | 0 |
| 8 | 64 | 5560.8 | 0 |
| 16 | 8 | 1130.9 | 0 |
| 16 | 32 | 5175.2 | 0 |
| 16 | 64 | 11090.8 | 0 |

Cost scales as expected O(K·D²) (the update is the dominant D² term; the
likelihood is an O(D) SIMD dot via `katgpt_types::simd::simd_dot_f32`).
The sub-µs gate point (K=8/D=8) passes with 1.8× headroom; larger grids
remain far under any per-entity tick budget (50 ms @ 20 Hz).

## T2.5 — identification quality (600 streams × L=24, seed 592, exact f64 generative sampling)

K Mess3 generators, one active per stream. Predictors: **(a)**
`NonergodicFilter` posterior; **(b)** single-belief leaky integrator at the
BEST of γ ∈ {0.8, 0.9, 0.95} (fair tuning; best was always γ=0.95);
**(c)** BoM-K unnormalized accumulation + hard `select_best` one-hot
prediction (1e-12 floor).

| K | metric | (a) filter | (b) leaky tuned | (c) BoM hard-sel |
|---:|---|---:|---:|---:|
| 2 | accuracy | **0.7650** | 0.7550 | **0.7650** |
| 2 | log-loss (nats) | **0.5049** | 0.5273 | 6.4933 |
| 2 | ECE (10-bin) | **0.0152** | 0.0626 | 0.2350 |
| 3 | accuracy | 0.5000 | **0.5050** | 0.5000 |
| 3 | log-loss (nats) | **0.9801** | 0.9849 | 13.8155 |
| 3 | ECE (10-bin) | **0.0645** | 0.0721 | 0.5000 |
| 4 | accuracy | 0.4067 | **0.4133** | 0.4067 |
| 4 | log-loss (nats) | **1.2473** | 1.2668 | 16.3944 |
| 4 | ECE (10-bin) | **0.0232** | 0.0567 | 0.5933 |

**Outcome vs the plan gate ("(a) beats both on log-loss at K=2..4")**: PASS
— (a) has the lowest log-loss at every K ∈ {2,3,4} against BOTH baselines.

Honest reading (Research 545 §5 demotion rule NOT triggered):

- **vs (b) leaky:** the win is calibration, not argmax — accuracy is a
  statistical tie (argmax flips on a handful of streams at K=3/4), but (a)
  wins log-loss at every K and ECE by 2–3×. As γ→1 the leaky baseline
  converges toward the exact filter, which is exactly why the tuned gap is
  thin at γ*=0.95: the baseline only works by approximating (a).
- **vs (c) BoM hard-select:** identical accuracy (same argmax — the
  accumulated unnormalized log-likelihood IS the same posterior math), but
  one-hot selection cannot express "not sure": 6.5–16.4 nats log-loss and
  0.24–0.59 ECE. This is the load-bearing argument for the two-level
  posterior over the existing BoM selection API: same decisions,
  principled confidence, and revival semantics BoM cannot represent.

## T2.6 — Report-the-Floor decision (Research 322 conformal-naive floor)

**Decision: the floor rule does NOT bind for this primitive as shipped.**
The conformal-naive floor gate applies to primitives claiming a probability
distribution over *future outcomes* — predictive intervals, quantiles,
coverage guarantees. `NonergodicFilter` exposes a *hypothesis posterior*
(`w` over which generator is active) plus per-generator inner beliefs — an
identification/commitment surface, not a forecast-interval surface. No
accessor returns an interval, quantile, or coverage claim, so there is no
UQ claim for the floor to police. **Trigger condition, documented:** if a
future consumer adds a predictive-interval or calibrated-probability
surface ON TOP of this filter (e.g. next-token intervals derived from
`telescope_into`), the conformal-naive floor
(`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340 m=1)
becomes a binding GOAT gate for that surface at its next re-gate, per the
grandfathered-primitive rule.

## T3.2 — promotion ruling

**STAYS OPT-IN.** `nonergodic_belief` is NOT added to any default feature
set. Rationale per AGENTS.md discipline and Plan 592 T3.2: default-on
requires a runtime-consumer GOAT (the conformal precedent — primitive-level
pass + consumer gates). The consumer wiring lives in riir-ai (Guide 373:
think-brain archetype tracking, P1/P2), which is out of scope for Plan 592.
This mirrors the `similarity_inference` demotion precedent: a
primitive-level GOAT pass with zero runtime consumers is not a promotion.

## Deviations from the plan (all recorded in the plan file)

1. **katgpt-types, not katgpt-core** (pre-assigned adaptation): SIMD dots
   via `katgpt_types::simd::simd_dot_f32` + `fast_sigmoid`; feature wired
   as a plain gate `nonergodic_belief = []` (katgpt-types is a
   non-optional dep).
2. **telescope_into takes `&mut [f32]`**, not `&mut [f32; K*D]` — stable
   Rust forbids `K * D` arithmetic in const positions
   (generic_const_exprs is unstable); length is debug-asserted.
3. **No criterion**: T2.3's "(criterion, release)" parenthetical replaced
   by the repo GOAT-bench convention (`harness = false` +
   `std::time::Instant` + inlined CountingAllocator — the canon_goat /
   procrustes_bench pattern). Criterion's harness would contaminate the
   alloc counting; this shape measures ns/tick AND allocs in one artifact.
4. **lib.rs module docs** gained one status line; the `.docs/09_feature_catalog`
   row and the `.research/545`/R248/R302 PASS-Redirect lines are DEFERRED
   to the coordinator (concurrent-agent write scope).

## Test counts (G3 evidence)

| Config | Result |
|---|---|
| `cargo test -p katgpt-micro-belief` (default) | 54 lib passed; gated targets compiled to nothing (0 passed rows are the expected `#![cfg]` green-zeros, all carrying `required-features` rows) |
| `cargo test -p katgpt-micro-belief --features nonergodic_belief` | **68 lib** (+14 nonergodic unit tests) + **9 G1 exactness** + **1 G4 alloc** + **1 G5 identification** = 79 |
| clippy (default / feature / all-features, --all-targets) | 0 warnings, 0 errors at all three |
