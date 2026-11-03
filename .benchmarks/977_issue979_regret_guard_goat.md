# Bench 977: regret_guard GOAT — adversary-aware serving gate (Issue 979)

**Date:** 2026-11-04
**Feature:** `regret_guard` (katgpt-core, zero new deps), root forwarder in the root manifest
**Issue:** [`.issues/979_regret_gap_gate_poc.md`](../.issues/979_regret_gap_gate_poc.md)
**Research:** [`.research/647_NTP_Regret_Robustification.md`](../.research/647_NTP_Regret_Robustification.md)
**Source:** arXiv:2603.28499 "Next-Token Prediction and Regret Minimization" (Mohri, Sanford, Schneider, Vodrahalli, Wu; ICML 2026) — Alg 1 + Thm 3.1, Alg 3 + §8, Thm 4.1/4.2, fetched and read in full this session
**Verdict: GOAT G1–G4 PASS — stays OPT-IN, promotion consumer-gated** (the gate is real when a lane shows the firing case with an incumbent-silent ground truth; candidate consumers recorded in Issue 979 T5)

## Substrate check (substrate-first, pre-implementation)

| Concept | Searched variants | Found |
|---|---|---|
| Regret gate / hedge / urn | `hedge`, `regret`, `polya`, `adversarial_stream`, `reference_leg`, `fallback_band` | `hint_regret` = a DIFFERENT concept (paired-rollout VoI for curriculum triage — stochastic returns, no adversary, no online sequence); `causal_id`'s "hedge" = the graph term (the issue's known collision). **Build new.** |
| Bandit regret (Plan 030) | stochastic pseudo-regret lanes | OPUS/EqR/BanditPruner track STOCHASTIC pseudo-regret — this issue's signal class (realized ADVERSARIAL regret vs a no-regret reference) has zero shipped consumers. Unchanged. |

Decision: new `regret_guard` module in katgpt-core; zero deps; zero `riir-*` deps; f64 accumulators + sequential folds (Issue-940); zero-alloc steady state.

## The η resolution (read-the-paper note)

The abs-page Alg-1 rendering shows `η = 1/T` in the QBR calls, but the
paper's own Lemma-2.3 PROOF uses `η = T^(−1/2)` → `λ_t = √T/(|Θ|+t−1)` and
the quantal price `ln|A|/√T` — exactly Issue 979's two pinned formulas
(the LaTeXML pseudocode rendering is the mangled half; the proof and the
band arithmetic carry it). Implemented: `qbr_eta` config knob, default
`1/√T`; deviating changes the Lem-2.3 constant, never the mechanism.

## The primitive

- `RegretGapGate` — Alg 1 as a runtime sticky-OR latch. Two legs scored on
  the same outcomes: frozen-leg QBR `π_f(a) ∝ exp(U(a,μ_t)/η)`; urn leg
  `μ_PU(θ) = (1+#{s<t: θ_s=θ})/(|Θ|+t−1)` whose QBR factors into
  exp-weights at `λ_t = 1/(η(|Θ|+t−1))` over realized utilities + a fixed
  prior — **the urn leg IS Hedge** (Lem 2.3). Both regrets share the
  per-action realized-utility vector, so the gap collapses to
  `(S_urn − S_frozen)/s`; the latch fires at
  `gap ≥ η·ln|A| + √(8(1+α)ln T/s)` (root over prefix length s, quantal
  price over T — the index pin) and never un-fires.
- `VDriftDetector` — the U-free binary escape (§8/Alg 3): V-shaped proper
  scores in their **piecewise-constant closed form** (the constant-p
  minimizer is `min{0, −vZ+(1−v)O, vZ+(v−1)O}` — brute-force
  spec-matched), threshold `B·√(ln(N_ε·t²/δ)/t)` with **B = 2 the V-loss
  range (load-bearing)** + the L_min warmup floor: at t = 1 the B-less
  bound is vacuous (a single round spans regret 2 — the probe caught the
  calibrated case firing at t = 1 pre-fix). **Binary only** — the |Θ| ≥ 4
  decomposition fails (Kleinberg et al.); the wider-alphabet surface does
  not exist by construction.
- `suffix_average_pullback` — Alg 2's indicator-form identity
  (`ln Z = (Δ − Σ ln π̄[a])/|A|`), roundtrip-tested through the QBR.

Loud refusals (fail-closed): no U (`UtilityMissing` — the U-free escape is
the V-detector only), malformed U (shape/range/non-finite), degenerate
alphabets, invalid config, malformed frozen dist (non-finite/negative/not
summing to 1), out-of-range outcome.

## G1 — the falsification battery (T3, all arms PASS)

| Arm | Result |
|---|---|
| deterministic-BR probe (Lem 2.2) | frozen regret **0.9995** (Ω(1) ✓) → gate fires at round **81** → served **0.158** (< 0.25 ✓, rescue ≥ 2× ✓) — pins the quantal leg as mandatory |
| 8-distribution battery, honest mode (frozen = oracle) | NO switch on all 8; served == frozen **bit-parity**; regrets **−0.12..−0.19** — the paper's Fig-1 negative-class shape (BERNOULLI ≈ −0.16) ✓ |
| 8-distribution battery, adversarial mode (counter-BR per shape) | ALL fire (rounds 81–191); frozen 0.50–1.0 → served **0.13–0.25** ✓ |
| TV-budget arm (Thm 3.1's ≤ \|A\|·T^(−α)) | **0 switches / 128** seeded in-distribution draws (bound 0.088) ✓ |
| de Bruijn certificate (Thm 4.1) | holds as the pre-registered **DISJUNCTION**: successor predictor on the complement stream regret **0.5 ≥ 1/24** (the un-escalated arm's red IS the theorem firing); constant predictors tie on the balanced stream (regret ≈ 0) and satisfy the **TV leg 0.50 ≥ 1/24** (first-bit disagreement = the data-processing witness) ✓ |
| Δ-context arm (Alg 2, staggered copies at horizon Δ) | **endpoint dominance: 0.319 (Δ=8) → 0.193 (Δ=128)**; mid-point 0.162 at Δ=32 — non-monotone middle recorded honestly (the staggered-restart interaction, not the theorem's 1/Δ term; the copies here have unbounded gate memory, unlike the paper's bounded-recall setting) |

T2 battery: the V-detector stays silent on the calibrated forecaster for
all 8 shapes and fires on the mirrored (counter-oracle) forecaster for all
8 ✓.

## G2 — ns/round (--release, 50k rounds, black_box'd)

| primitive | ns/round | complexity |
|---|---|---|
| RegretGapGate 2×2 indicator | **18.8** | O(\|A\|·\|Θ\|) = 4 |
| RegretGapGate 4×3 general U | **31.9** | 12 |
| VDriftDetector ε=0.05 (21 tips) | **21.5** | O(N_ε) |

Bar: µs-class per round (the issue's G2) → **30–53× under**. Timed regions
run through `tests/common/ab_timing.rs::best_of_us` (the Issue-855
treatment — the loud-zero defence the timed-region guard gate requires;
the first hand-rolled-Instant draft of this bench was caught mid-flight by
a unit-conversion slip, which is exactly the class the shared harness
exists to police).

**Box state (Issue-021 law):** macOS arm64 (amo), 30 logical CPUs, AC
power, lowpowermode 0 (Automatic), load 3.22/3.60/3.49 at bench time
(01:27 local, no cargo compute consumers at snapshot). Isolated target
dir `/tmp/issue980_tgt`.

## G3 — untouched-surface discipline

- Default-feature list unchanged (opt-in only); default `cargo test -p
  katgpt-core --lib` **2167 passed, 0 failed** (the pre-979 count)
- Flag-off posture: without the feature the gate targets REFUSE
  (`required-features` + whole-file `#![cfg]` — the green-zero pairing,
  never a silent `0 passed`)
- `--no-default-features --features regret_guard` clippy clean;
  `--all-features` compiles clean
- Release-mode battery: all 9 tests pass at `--release`

## G4 — zero-alloc steady state (canary'd)

`observe()` rounds (both primitives), construction excluded, **0 allocs**
over 50,000 rounds, after the Issue-714 liveness canary proves the counter
is live.

## Honest findings (recorded, not buried)

1. **The B constant is load-bearing**: the issue's `√(log(N_ε t²/δ)/t)`
   threshold has no loss-range constant; at t = 1 the bound (1.23) is
   smaller than a single round's regret span (2) — the calibrated fixture
   FIRED at t = 1 pre-fix. Shipped `B·√(...)` with B = 2 (the V-loss
   range) + the L_min warmup floor (the GainCostLoopHalter law). The
   paper's Lemma-8.8 constant `c_B` carries the same weight.
2. **Δ-arm non-monotone middle**: endpoint dominance holds (pre-registered
   assertion), the Δ=32 mid-point dips lowest. The realization's copies
   carry unbounded gate memory (vs the paper's bounded-recall AverageRestartHedge)
   — recorded as the known deviation; a faithful bounded-recall port is
   consumer-gated work.
3. **Leg-TV diagnostic ≠ served fidelity**: the gate's
   `mean_next_token_tv` measures frozen-vs-urn distance (informative,
   recorded), never served-vs-frozen — the fidelity assertion uses
   bit-parity of the served and frozen legs when unswitched (the
   strongest form; d_NT over full-sequence TV per the 0.42 same-seed law).

## Promotion verdict

**GOAT G1–G4 PASS; STAYS OPT-IN, promotion consumer-gated.** The T5
candidates (reflex third fused-abstain signal / rethink ESC flag / seal
duel-PvP lane) each need their own repo's gate + a documented firing case
with incumbent-silent ground truth; the public comparison-lane change is
owner-gated per reflex's MOAT. The certificate is pre-flight (existence
over a family at a context width); the gate is runtime (being-beaten-now)
— never conflate.

## Artifacts

- Module: `crates/katgpt-core/src/regret_guard.rs` (10 unit tests)
- Gate battery: `crates/katgpt-core/tests/issue979_regret_guard_gates.rs` (9 tests, green-zero paired with `required-features`)
- Bench: `crates/katgpt-core/benches/issue979_regret_guard_goat.rs`
- Feature catalog: `.docs/09_feature_catalog/opt_in_features.md` + README row

## Reproduce

```bash
cargo test -p katgpt-core --features regret_guard --test issue979_regret_guard_gates -- --nocapture
cargo test -p katgpt-core --features regret_guard --lib regret_guard
cargo bench -p katgpt-core --features regret_guard --bench issue979_regret_guard_goat
```
