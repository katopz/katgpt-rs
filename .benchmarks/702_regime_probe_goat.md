# Bench 702 — Regime-Probe Primitives GOAT (Issue 740, Research 541)

**Status:** Complete — G1 PASS · G2 PASS (with recorded caveats + a scope
boundary) · G3 PASS · G4 PASS — feature stays OPT-IN (no-default-consumer
rule)
**Date:** 2026-09-09
**Issue:** [`740_regime_probe_primitives.md`](../.issues/740_regime_probe_primitives.md)
**Research:** [`541_UDDM_Associative_Memory_Regime_Probes.md`](../.research/541_UDDM_Associative_Memory_Regime_Probes.md)
**Source:** arXiv:2604.26841 (EMNLP 2026) — UDDMs as associative memories
**Gate binary:** `crates/katgpt-core/tests/bench_702_regime_probe_goat.rs`
(`cargo test -p katgpt-core --features regime_probe,hebbian_kernel_memory
--test bench_702_regime_probe_goat -- --nocapture`)
**Golden vectors:** `crates/katgpt-core/tests/regime_probe_golden.rs`
**Alloc audit:** `crates/katgpt-core/tests/regime_probe_alloc_check.rs`

## What shipped (T1–T4)

`crates/katgpt-core/src/regime_probe/` behind `regime_probe` (opt-in; root
forwards `regime_probe = ["katgpt-core/regime_probe"]`):

| Task | Surface | Notes |
|---|---|---|
| T1 | `entropy::conditional_entropy_nats` (+ `_into` batch, `mean_…`) | `H = ln Z − E_p[x−max]` via the SHARED kernel `simd::logsumexp_parts` — factored from `breakeven/fidelity.rs::cross_entropy` (the issue's reuse mandate; the existing caller's outputs are bit-identical, it just ignores the new third return value). |
| T2 | `gap::entropy_gap` (+ `_into`) | Two-sample mean gap + Cohen's-d (f64 accumulation, f32 storage); BLAKE3 artifact over a canonical LE encoding (`KRPG`); report retains samples for auditability. |
| T3 | `basin::basin_probe` (+ `_into`), trait `FrozenRenovator` | eq-12 protocol: seeded Fisher–Yates corruption (fastrand `with_seed` — the `data_probe::markov` RNG convention; deterministic) → sequential argmax renovation over corrupted sites only → recovery rate + overlap; BLAKE3 artifact (`KRPB`). |
| T4 | `gardner::{gamma_capacity, kappa_max, basin_radius_bound, phi_cdf, phi_pdf}` | `1/γ_c = (1+κ²)Φ(κ) + κφ(κ)`; Φ via A&S 7.1.5 all-positive series (no cancellation, f64); inversion computed ONCE into a `OnceLock` uniform-log-γ grid (4096 pts) + 3-point quadratic interpolation, O(1)/query, zero per-call alloc. |

RNG substrate note: the probes use `fastrand::Rng::with_seed` (the
`data_probe::markov` harness convention — a katgpt-core dep, seedable,
platform-stable). `hebbian_kernel_memory::SeedRng` drives the G2 memory
construction + the GOAT fixtures. No new deps anywhere (blake3 already
non-optional).

## G1 — discriminative validity (T5): PASS

Constructed pair, zero training: an exact-LUT **memorizer** (capacity 128
first-seen contexts, sharpened empirical distributions, uniform on miss) vs a
kernel-smoothed **generalizer** (structure-pooled Dirichlet smoothing over
sum-mod-V classes), both consuming the SAME seeded corpus from a fixed world
(V=8, 3-token contexts, L=64, near-one-hot class table).

Detector arm: reference = training corpus; generated = fresh deployment-time
inputs (held-out world draws). Secondary arm (paper's model-sampled sequences)
printed, not gated — a greedy constructed memorizer keeps its own samples on
memorized mode-paths (measured gap 0.23→0.10 nats, same sign, smaller).

| Load N | distinct ctx | stored | gap_mem (nats) | gap_gen (nats) | rec_mem_train | rec_gen_train |
|---|---|---|---|---|---|---|
| 2 | 101 | 101/128 | **+1.376** | +0.004 | **0.750** | 0.438 |
| 8 | 281 | 128 (capped) | +0.230 | +0.011 | 0.438 | 0.500 |
| 32 | 467 | 128 (capped) | **−0.006** (converged) | −0.006 | **0.125** | **0.375** |

- **Separation at low load**: memorizer gap 1.376 nats vs generalizer 0.004
  (asserted > 1.0 and > +0.5 margin). ✓
- **Falling arm + convergence**: memorizer gap 1.376 → −0.006 across the
  capacity crossing; train-recovery 0.750 → 0.438 → 0.125 monotone. ✓
- **Fig 1B crossover**: at high load the generalizer renovates training data
  3× better than the memorizer (0.375 vs 0.125, asserted ≥ +0.2). ✓
- **Regime classification** (gap threshold 0.5 nat): mem@low → Memorizing,
  gen@low → Generalized, mem@high → Generalized (post-transition). ✓

## G2 — bound-holds (T6): PASS, with recorded caveats

Constructed `HebbianKernelMemory<64>` consumer (D=64, m=512, ±1 keys/values,
Whitened-vs-Unwhitened ablation, loads γ = F/m ∈ {1, 4}), renovated via a
Hebbian-backed single-site renovator (two completions scored against the value
table; Bernoulli posterior = `sigmoid((s₁−s₀)/σ)` — sigmoid, not a multi-way
softmax gate). ρ_c = largest exact-bit grid ρ = k/64 with overlap ≥ 0.99,
median over 4 keys. κ_achieved = worst z-scored decoding margin over 64
sampled keys (per-key score-spread normalization).

| Load | variant | κ_achieved | bound (κ/2)² | per-key ρ_c | median ρ_c | holds |
|---|---|---|---|---|---|---|
| γ=1 (F=512) | Unwhitened | 0.37 | 0.035 | 0.031 / 0.047 / 0.094 / 0.188 | **0.094** | ✓ |
| γ=4 (F=2048) | Unwhitened | −2.71 | 0.000 | 0.000 / 0.000 / 0.016 / 0.078 | **0.016** | ✓ (bound degenerate) |

Load monotonicity: median ρ_c falls 0.094 → 0.016 with load ✓. Single-bit
restoration failures (basin existence premise): **3/64 sites** at γ=1,
asserted ≤ 10%. ✓

**Caveats recorded with the PASS:**

1. **Weakest-key pairing**: at γ=1 the weakest sampled key measures ρ_c=0.031
   against its own bound 0.035 — one grid notch (1/64 = 0.0156) below, i.e.
   below measurement resolution. The gate's pairing is memory-level κ (γ_min,
   the standard decoding-margin definition) vs pattern-level median ρ_c
   (the task's literal reading); the stricter per-key pairing is
   sub-resolution indeterminate.
2. **Negative margins at γ=4**: some keys carry κ < 0 — past effective
   retrieval capacity. The bound degenerates to 0 and "holds" trivially. The
   consumer signal: require `κ_achieved > 2` before quoting any tolerance
   (the M4 capacity-planner read).
3. **Scope boundary (the important one)**: the **Whitened** interpolant
   variant FAILS the bound's premise outright. Measured at γ=1 (F=m=512,
   maximally ill-conditioned whitening): ⟨v₀, MLP(k₀)⟩ = 64.28 at the key,
   but ONE bit flip drives max_v·MLP to **865.5** (13×) — the least-squares
   readout amplifies off-key perturbations and every key sits on a needle
   basin (0/64 single-bit sites restorable; ρ_c = 0 at ρ = 1/64). The CLT
   premise (flip perturbation ≈ 2√ρ in margin units) does not describe the
   interpolating readout. The Gardner bound is a **Hebbian-correlator**
   (Unwhitened) result on this substrate — the paper's own storage model —
   and NOT a readout-family result.

## G3 — bit-determinism (T7): PASS

Full G1 measurement path and the G2 basin path rerun from fresh objects with
identical seeds: gap reports content-equal, BLAKE3 artifacts bit-identical
(`g3_bit_determinism_across_fresh_reruns`). Same for the golden T2/T3
artifact gates (×2 runs, incl. scratch capacity-reuse).

## G4 — zero-alloc steady state (T7): PASS

`regime_probe_alloc_check` (separate binary, CountingAllocator convention):
after warmup, 1000 × `conditional_entropies_into` (256×32) + 100 ×
`entropy_gap_into` + 100 × `basin_probe_into` allocate **0 bytes**. The
scratch/report capacity-reuse protocol holds; the canonical encodings hash
incrementally (no byte buffer).

## T4 golden vectors

LUT vs direct bisection inversion of the same closed forms, 2000-point sweep
over γ ∈ [0.031, 2.0) (incommensurate step): **worst κ_max rel err 5.20e-10,
worst ρ_bound rel err 1.04e-9** (gate: < 1e-6 — 400× margin). Self-check:
`γ_c(κ_max_bisection(γ)) = γ` to 1e-9. Known values: γ_c(0) = 2 exact,
κ_max(1.0) = 0.4712, Φ known answers to 1e-9. Implementation notes: 2048
points on a uniform-κ grid measured 9.7e-4 (flat tail); uniform-log-γ +
linear measured 1.0e-6 at the κ→0 end (constant κ'' ≈ π/2 vs vanishing κ —
relative error grows like 1/κ); the 3-point quadratic closes it at 5.2e-10.

## Test counts

| Suite | Result |
|---|---|
| `cargo test -p katgpt-core --features regime_probe --lib` | **2001 passed**, 0 failed (22 new module unit tests) |
| `cargo test -p katgpt-core --lib` (default) | **1979 passed**, 0 failed (no regression: 2001 − 22 = 1979) |
| `--test regime_probe_golden` | 10 passed |
| `--test regime_probe_alloc_check` | 1 passed |
| `--test bench_702_regime_probe_goat` | 4 passed |
| `cargo test -p katgpt-types --lib` (simd kernel touched) | 132 passed |
| `cargo clippy -p katgpt-core --features regime_probe --all-targets -- -D warnings` | clean |
| `cargo clippy -p katgpt-core --all-targets -- -D warnings` | clean |
| `python3 scripts/count_features.py` | all claim sites match (580 total / 196 default-on) |

Environment note: run under `CARGO_TARGET_DIR=/tmp/regime_probe` (shared
worktree; a sibling session holds `target/` busy). Debug-profile numbers
throughout; the GOAT gates here are discriminative/deterministic, not latency.

## UQ floor (T9)

The probes are **bare detectors** — they emit raw scalars (nats, gaps,
recovery rates, ρ bounds), not calibrated probabilities or intervals — so the
conformal-naive floor rule (Research 322 / Plan 340) is NOT triggered. **This
exemption must be re-affirmed at promotion time if the probe ever emits a
calibrated probability/interval** (e.g. if a consumer starts treating
`mean_gap` as a calibrated memorization probability or the renovator
posteriors as UQ outputs); at that point the primitive must beat
`ConformalIntervalCalibrator<SeasonalNaiveForecaster>` on CRPS/coverage/Winkler.

## Consumer addendum (2026-09-09): first real measurement — riir-clippy run #84

The first consumer landed: riir-clippy Issue 077 T1+T4 (`1994a8a`+`ffdd7a9`)
wires `katgpt-core::regime_probe` into its score_bench (`score_bench`
feature forwards `katgpt-core/regime_probe`; new `src/score_bench/ood.rs`;
additive JSONL field; no parallel entropy implementation — the boundary
mandate held). Measured on a real `--score-bench` run (#84, 35 fixtures, heal
96.2%): reference = v1 frozen (35) vs OOD-labeled = gen6 real-repo (17):

| mean_ref | mean_ood | gap | Cohen's d | classification |
|---|---|---|---|---|
| 3.840 nats | 3.110 nats | **−0.729** | −1.11 | **anomalous_negative** |

Artifact `faa021c8…` (id-sorted, bit-deterministic). The honest reading: the
OOD-labeled set fires MORE sharply than the reference — gen6 is carved from
the SAME pre-heal sibling states as the seed corpus, so **provenance family,
not the held-out label, decided the gap side**. This is the axis working
(d it discriminated, strongly) while the convergent-validity claim (077's
GOAT: regime boundary agrees with an external oracle inflection) is NOT yet
demonstrated — the label coupling poisons the pairing. Both interpretations
are documented in riir-clippy `.docs/08_benchmarks/entropy_gap_ood_axis.md`.

## Promotion status (updated by the addendum)

`regime_probe` stays **opt-in**. Requirement (a) — a consumer — is now MET
(riir-clippy score_bench), but the consumer's first real-corpus measurement
is anomalous_negative, so promotion to default is STILL DEFERRED. Unblock: a
reference/OOD pairing with disjoint provenance families (record provenance
next to the frozen labels in score_bench fixtures), then re-read this record
plus 077's convergent-validity GOAT.

## What remains

- The engine serving-health audit remains an OPTIONAL second consumer (the
  077 score_bench axis satisfies T8's "and/or").
- The paper's model-sampled ("self-entropy") arm is recorded as a secondary
  signal (M6); a consumer wanting the paper's exact protocol gates on it.
- KS statistic for the two-sample detector: skipped (mean gap + Cohen's d
  carried the G1 separation at 300× the threshold); add only if a consumer
  needs distribution-shape sensitivity.
