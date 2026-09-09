# Issue 740: Regime-Probe Primitives — Entropy Gap, Basin Probe, Gardner LUT (arXiv:2604.26841)

**Status:** Implemented — T1–T7, T9 landed; GOAT G1/G3/G4 PASS, G2 PASS with
recorded caveats (Bench 702); feature stays opt-in; T8 consumer half
(riir-clippy score_bench) remains
**Date:** 2026-09-08
**Research:** [katgpt-rs/.research/541_UDDM_Associative_Memory_Regime_Probes.md](../.research/541_UDDM_Associative_Memory_Regime_Probes.md) @ katgpt-rs `77169275`
**Source:** [arXiv:2604.26841](https://arxiv.org/abs/2604.26841) — UDDMs as associative memories; conditional entropy as a training-free memorization↔generalization probe.

## Ask

Ship the predictor-agnostic regime-probe suite as modelless katgpt-core primitives (rows M2/M3/M4 of Research 541 §2). Every probe consumes only output categoricals a serving path already produces; ground-truth validation uses deterministically constructed predictors (regime known by construction — zero training).

## Tasks

- [x] **T1** New module `crates/katgpt-core/src/regime_probe/` behind feature flag `regime_probe` (opt-in): per-position conditional entropy of a categorical (max-shift + logsumexp, zero-alloc, `f32`); reuses the `cross_entropy` kernel shape from `breakeven/fidelity.rs` — do not duplicate, factor or delegate. *(Factored into `katgpt-types::simd::logsumexp_parts` — one kernel shape; `cross_entropy` delegates bit-identically and ignores the new third return value.)*
- [x] **T2** Entropy-gap detector (M2): two-sample statistic (mean gap first; KS optional) between reference-corpus and generated-sequence entropy distributions; bit-deterministic, BLAKE3-able artifact output. Sibling of `data_probe/markov.rs` in harness style. *(KS skipped — mean gap + Cohen's d carried the G1 separation at 300× threshold; documented in Bench 702.)*
- [x] **T3** Corrupt-recovery basin probe (M3): corrupt fraction ρ of positions → renovate via a caller-supplied frozen renovator (trait seam shaped like `ugc_schedule::UgcDenoiser`) → recovery rate on corrupted positions (eq 12 protocol).
- [x] **T4** Gardner capacity LUT (M4): `κ_max(γ)` from `1/γ_c(κ) = (1+κ²)Φ(κ) + κφ(κ)` inverted numerically at build time; expose `basin_radius_bound(gamma) -> rho` via `κ > 2√ρ`; golden vectors vs direct inversion (rel-err < 1e-6). *(OnceLock uniform-log-γ grid + quadratic interpolation, computed once at first use — "build time" read as computed-once; measured worst err 5.2e-10.)*
- [x] **T5** GOAT G1 (discriminative validity): probes separate a constructed exact-LUT "memorizer" vs kernel-smoothed "generalizer" across a load sweep — transition must reproduce the paper's Fig 1B cross-over shape. **PASS** — gap 1.376→−0.006 nats across the capacity crossing; train-recovery 0.750→0.125 monotone; generalizer crossover at high load (0.375 vs 0.125); classification labels all three regime points correctly.
- [x] **T6** GOAT G2 (bound-holds): flip-fraction `ρ_c` at 0.99-overlap on a constructed Hebbian memory (`hebbian_kernel_memory` consumer) sits above the `κ_achieved`-derived bound across loads (Fig 6 shape). **PASS with caveats** on the Unwhitened (Hebbian-correlator) variant: measured median ρ_c 0.094 ≥ bound 0.035 at γ=1; 0.016 ≥ 0.000 at γ=4 (bound degenerate — margins go negative past effective capacity). Recorded scope boundary: the **Whitened interpolant fails the CLT premise outright** (needle basins — one-bit flip drives max_v·MLP 64→865; 0/64 single-bit sites restorable) — the bound is a Hebbian-correlator result, not a readout-family result.
- [x] **T7** G3/G4: bit-determinism under fixed seed; zero-alloc scratch-buffer protocol; `cargo clippy -D warnings` clean at default + `--features regime_probe`. **PASS** — artifacts bit-identical across fresh reruns; 0 bytes steady-state (1000 entropy batches + 100 gap + 100 basin probes); clippy clean both feature sets.
- [-] **T8** Consumers wired + measured: `riir-clippy` score_bench OOD axis (its Issue 077) and/or engine serving-health audit; one benchmark doc in `.benchmarks/` with the G1/G2 tables before any promotion talk. *(The `.benchmarks/702_regime_probe_goat.md` half is DONE. The consumer half is other-repos work — riir-clippy Issue 077 + engine serving-health audit — deliberately not touched from this repo; it remains the open half of T8.)*
- [x] **T9** UQ floor check (conditional): if the probe ever emits calibrated probabilities/intervals, benchmark against the conformal-naive floor (Research 322 / Plan 340) — bare detectors are exempt but this must be re-affirmed at promotion time. *(Exemption recorded in Bench 702 §"UQ floor (T9)"; re-affirmation note in place.)*

## Notes

- Do NOT reopen Plan 276's `AttractorKernel` speculatively; Research 541 §3 records why it failed and what would unblock it (Hebbian construction à la R455) if a consumer materializes.
- Raw-scalar outputs only (entropy nats, gap, ρ bound) — latent read → scalar out, the sanctioned bridge direction.
- Measured record: [`.benchmarks/702_regime_probe_goat.md`](../.benchmarks/702_regime_probe_goat.md). Implementation commit: `781264aa` (feat: regime-probe primitives).
