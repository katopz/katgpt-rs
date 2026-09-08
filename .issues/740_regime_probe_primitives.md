# Issue 740: Regime-Probe Primitives — Entropy Gap, Basin Probe, Gardner LUT (arXiv:2604.26841)

**Status:** Open — POC task, feature-gated
**Date:** 2026-09-08
**Research:** [katgpt-rs/.research/541_UDDM_Associative_Memory_Regime_Probes.md](../.research/541_UDDM_Associative_Memory_Regime_Probes.md) @ katgpt-rs `77169275`
**Source:** [arXiv:2604.26841](https://arxiv.org/abs/2604.26841) — UDDMs as associative memories; conditional entropy as a training-free memorization↔generalization probe.

## Ask

Ship the predictor-agnostic regime-probe suite as modelless katgpt-core primitives (rows M2/M3/M4 of Research 541 §2). Every probe consumes only output categoricals a serving path already produces; ground-truth validation uses deterministically constructed predictors (regime known by construction — zero training).

## Tasks

- [ ] **T1** New module `crates/katgpt-core/src/regime_probe/` behind feature flag `regime_probe` (opt-in): per-position conditional entropy of a categorical (max-shift + logsumexp, zero-alloc, `f32`); reuses the `cross_entropy` kernel shape from `breakeven/fidelity.rs` — do not duplicate, factor or delegate.
- [ ] **T2** Entropy-gap detector (M2): two-sample statistic (mean gap first; KS optional) between reference-corpus and generated-sequence entropy distributions; bit-deterministic, BLAKE3-able artifact output. Sibling of `data_probe/markov.rs` in harness style.
- [ ] **T3** Corrupt-recovery basin probe (M3): corrupt fraction ρ of positions → renovate via a caller-supplied frozen renovator (trait seam shaped like `ugc_schedule::UgcDenoiser`) → recovery rate on corrupted positions (eq 12 protocol).
- [ ] **T4** Gardner capacity LUT (M4): `κ_max(γ)` from `1/γ_c(κ) = (1+κ²)Φ(κ) + κφ(κ)` inverted numerically at build time; expose `basin_radius_bound(gamma) -> rho` via `κ > 2√ρ`; golden vectors vs direct inversion (rel-err < 1e-6).
- [ ] **T5** GOAT G1 (discriminative validity): probes separate a constructed exact-LUT "memorizer" vs kernel-smoothed "generalizer" across a load sweep — transition must reproduce the paper's Fig 1B cross-over shape.
- [ ] **T6** GOAT G2 (bound-holds): flip-fraction `ρ_c` at 0.99-overlap on a constructed Hebbian memory (`hebbian_kernel_memory` consumer) sits above the `κ_achieved`-derived bound across loads (Fig 6 shape).
- [ ] **T7** G3/G4: bit-determinism under fixed seed; zero-alloc scratch-buffer protocol; `cargo clippy -D warnings` clean at default + `--features regime_probe`.
- [ ] **T8** Consumers wired + measured: `riir-clippy` score_bench OOD axis (its Issue 077) and/or engine serving-health audit; one benchmark doc in `.benchmarks/` with the G1/G2 tables before any promotion talk.
- [ ] **T9** UQ floor check (conditional): if the probe ever emits calibrated probabilities/intervals, benchmark against the conformal-naive floor (Research 322 / Plan 340) — bare detectors are exempt but this must be re-affirmed at promotion time.

## Notes

- Do NOT reopen Plan 276's `AttractorKernel` speculatively; Research 541 §3 records why it failed and what would unblock it (Hebbian construction à la R455) if a consumer materializes.
- Raw-scalar outputs only (entropy nats, gap, ρ bound) — latent read → scalar out, the sanctioned bridge direction.
