# Bench 703 — QSG Gossip Physics GOAT Gate (Plan 589 / Research 542)

**Date:** 2026-09-10
**Feature:** `qsg_gossip` (opt-in, `katgpt-core`)
**Source:** arXiv:2603.24676 (Tanaka, Mar 2026) — Quantized Simplex Gossip
**Targets:** `cargo test -p katgpt-core --features qsg_gossip --lib qsg` (21 tests) ·
`cargo test -p katgpt-core --features qsg_gossip --test bench_703_qsg_scaling_laws --release` (6 gates) ·
`cargo test -p katgpt-core --features qsg_gossip --test qsg_gossip_alloc_check` (1 gate)

**Verdict: GOAT — G1–G4 ALL PASS.** Opt-in; promotion waits on a production
consumer (riir-ai Issue 907; the signed_coupling precedent).

---

## G1 — Correctness: the kernel IS the law

### In-module identity pins (21 tests, dev profile, 5.0 s)

| Law | Test | Result |
|---|---|---|
| Thm 1 (Hard variance injection) | `theorem1_hard_variance_injection_identity` | PASS (paired estimator, ≤1%) |
| Thm 2 (TopM bandwidth, m ∈ {1,2,3,5,10}) | `theorem2_topm_bandwidth_identity` | PASS (paired, ≤1.5%) |
| Symmetric 1/m corollary (paper Fig. 6c) | `symmetric_1_over_m_corollary` | PASS (≤3%) |
| Soft martingale `E[x̄'|X] = x̄` | `soft_exchange_preserves_the_mean` | PASS (20k pairs, per-coord) |
| Soft contracts V | `soft_exchange_contracts_disagreement` | PASS (measured −5.03e-2 vs −5.06e-2) |
| α=1 voter reduction, winner prob = x̄ₖ(0) | `alpha_one_reduces_to_uniform_winner_voter_model` | PASS (1500 runs, |f − ⅓| < 0.05) |
| Soft from symmetry never breaks it | `soft_from_symmetry_never_breaks_it` | PASS (exact, 500 steps) |

Plus kernel units: config validation, categorical CDF walk (strict-< tie
rule), message modes (Hard one-hot / TopM empirical / Soft copy / TopM(0)
degrade), `uniform_ordered_pair` coverage + determinism, hand-computed blend
step with U before/after (½ → 5/9), reducer degeneracies, simplex
normalization (clamp / reset / idempotence).

**Provenance note (load-bearing):** the scraped PDF renders the Soft V-
contraction as `(1−α+αN)`; the α→1 limit disambiguates — that form would
collapse V in one step, but α=1 is the voter model (consensus ~N² steps), so
the per-step contraction must be O(1/N²)-class. The corrected identity is
`E[ΔV|X]_soft = −(2α/(N−1))·(1−α+α/N)·V` — the kernel matches it to <1%
(measured −4.898e-2/−5.030e-2 vs theory −5.062e-2 at 12k/60k pairs), and the
same PDF-mangling class hits `Γh` in the main text (the A.4 diffusion + the
`Nc ~ α/(m|h|)` crossover fix α in the denominator). Same scrape-loss class
as R296's vocabulary lesson: never trust a scraped equation against its own
limiting case.

### Physics-law gates (`bench_703_qsg_scaling_laws`, RELEASE profile)

| Gate | Result |
|---|---|
| G1a consensus time ∝ N² (N ∈ {4…32}, 7 trials, log-log fit) | PASS — slope within [1.85, 2.15], R² ≥ 0.98 |
| G1b per-step drift = α²/(mN²)(1−1/K) (reset-per-draw, N ∈ {8…64}) | PASS — ≤1% per N; drift(8)/drift(32) = 16 ± 0.32 |
| G1b 1/m law (m ∈ {1,2,4,8}) | PASS — ratios ≤2%/m |
| G1c trajectory vs exact two-moment (U,V) flow (paper Eq. 31/33 corrected; N=24, K=10, α=0.2; ensemble of 48 runs, 3 checkpoints to 2.5·t_char) | PASS — E[U] within 0.08, E[V] within 0.15 at every checkpoint |
| G1d fixation collapse onto Γh = mNh/α | PASS — see below |
| G1e tempered crossover ΓT = (mN/α)\|1/T−1\| | PASS — T=0.5 amplifies to U > 0.9; T=2.0 damps (U < 0.45); gap > 0.35 |

### The Γh collapse (the paper's headline) + one honest deviation

Measured Pr(fix) (K=2, α=0.5, m=1, U⋆=0.9 deep cut, 300 trials/point):

| Γh | (N, h) points | measured | σ(Γh) | σ(Γh/2) |
|---|---|---|---|---|
| 0.08 | (8, 0.005) | 0.487 | 0.520 | 0.510 |
| 0.80 | (8, 0.05) / (64, 0.00625) | 0.617 / 0.580 | 0.690 | 0.599 |
| 3.20 | (8, 0.2) / (16, 0.1) / (32, 0.05) / (64, 0.025) | 0.843 / 0.790 / 0.823 / 0.800 | 0.961 | 0.832 |
| 6.40 | (8, 0.4) | 0.967 | 0.998 | 0.961 |

- **Collapse: PASS** — same-Γh pairs agree to ≤0.043 across an 8× population
  range (the paper's single-parameter claim, gate tolerance 0.08).
- **Deviation, recorded:** the kernel's curve is σ(Γh/2), not the first-order
  diffusion's σ(Γh) — max deviation 0.043 vs σ(Γh/2), 0.174 vs σ(Γh).
  Mechanism: the first-order diffusion drops the pair-choice heterogeneity
  variance (WHICH listener hears the message), which doubles the effective
  noise and halves the logistic's slope. The direction check still holds
  (Pr(N=64, h=0.02) > Pr(N=8, h=0.02) + 0.1): larger N suppresses drift and
  makes the same bias more decisive.

### Method notes (for the next re-gate)

- One-step identities use **reset-per-draw** estimators (the paper's Fig. 6b
  shared-snapshot protocol). Trajectory-window drift estimates are contaminated
  by U's martingale random walk (std ~ (2α/N)·0.1·√t) — do not use them for
  the drift laws (it cost a false N=16 failure to learn this).
- The two-moment flow is EXACT for E[U_t], E[V_t] (the moment map is linear in
  (U,V)), so ensemble means converge to it; 48 runs put the sampling error at
  ~0.02 (per-run martingale std ≈ 0.14 at t_char). Single runs may not be
  compared to the flow at long horizons.
- Per-run uniform budgets need **cursor tracking** across checkpoints — a fresh
  stream per checkpoint silently replays the same draws and corrupts the
  ensemble (found via a 4σ E[U] deviation that vanished once cursors were
  tracked).

## G2 — Perf: tick-path primitive

10⁶ Hard interactions, K=8, N=1024, release profile:
**20.6 ms (20.5–20.8 ns/interaction)** — 4.9× inside the ≤100 ms gate bound,
same class as signed_coupling's 1.8 ns/edge (this kernel samples + blends a
K-vector per interaction, so ~10× per-interaction cost is expected).
Debug-profile smoke bound: 8 s (the gate bound is a release-profile claim).
No GPU exclusivity needed (CPU scalar kernel; sibling cargo builds do not
invalidate ns-class scalar timings at this margin).

## G3 — No regression

`cargo clippy -p katgpt-core --all-targets -- -D warnings` (feature OFF):
clean — the module compiles to nothing at default features; both integration
test targets are `required-features`-gated + `#![cfg]`-gated (both defenses
per the Issue 713 rule), so default `cargo test` never compiles them.

## G4 — Alloc-free

`qsg_gossip_alloc_check`: warm tick outside the window, `assert_counter_is_live`
canary (Issue 714), then 1000 TopM(3) steps + mean/U/V/uncertainty reducers on
caller scratch: **0 allocations** (per-thread counter).

## Counts (floors for the next re-gate — a drop means the feature set broke)

| Target | Floor |
|---|---|
| `--lib qsg` (feature on) | 21 passed |
| `bench_703_qsg_scaling_laws` (release) | 6 passed |
| `qsg_gossip_alloc_check` | 1 passed |

## Slot ledger

Crowd-dynamics family: `signed_coupling` (temperature T axis) · `qsg_gossip`
(bandwidth m + adaptation α axes) · `mean_field` (order-parameter reducers) —
siblings, none demoted; QSG opens a new axis rather than replacing a slot.
Feature stays opt-in until the riir-ai consumer lands; re-gate on any touch.
