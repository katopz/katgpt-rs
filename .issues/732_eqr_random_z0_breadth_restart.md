# Issue 732 — Fresh-z₀ breadth-restart arm + D-first law for best_of_k_rollouts (EqR RI axis)

**Status:** OPEN — bench-first; all arms behind `eqr_convergence`; promote only on GOAT with the negative control intact.
**Date:** 2026-09-06
**Source:** arXiv:2605.21488 re-audit addendum ([`.research/079_EqR_Equilibrium_Reasoners.md`](../.research/079_EqR_Equilibrium_Reasoners.md) §10); cross-repo substrate sweep 2026-09-06.
**Lineage:** Plan 119 (selection ✅) — this is the **breadth-initialization** half.

## Gap

`best_of_k_rollouts` breadth today = per-rollout SDE noise (γ ∈ {0.5, 1.0}) around the SAME base state — low-variance perturbation inside one basin. EqR's randomized-initialization axis at inference = **independent restarts from fresh random z₀** (Gaussian, large σ) probing DIFFERENT basins, aggregated by `Top1Converged`. The paper: breadth useless below depth D ≳ 4 (interaction law); `Top1Converged` beats majority vote ONLY on shaped landscapes (on unshaped ones it can lose — convergence certifies basin membership, not correctness); restart-consistency Δ_PI(B) quantifies path independence (zero code today; research note 344 discusses the concept only).

Companion deltas from the full text worth measuring while the bench is open:

1. **D-first interaction law** — re-measure the knee on our loop; if it differs from D≳4, the paper's constant is refuted on our maps and the scheduler re-pins to the measured value.
2. **Negative control (pre-registered)** — on unshaped fixtures `Top1Converged` must NOT beat `MostFrequent`; a win indicts the fixture, not the method.
3. **Four-mode landscape taxonomy** — no-correct-attractor / correct+spurious / narrow-basin / aligned → which axis to spend (neither / breadth / breadth+depth / depth). Label-free proxies: residual plateau test + cross-restart decode clustering. No single classifier ships anywhere in the workspace (cousins: Churning verdict, `intrinsic_grounded_gap` spurious-attractor check, bench_011 5-mode trajectory taxonomy).
4. **Δ_PI restart-consistency metric** — E[|mean acc over B restarts − single-run acc|].

## Tasks

- [ ] T1 — `restart_mode` knob (`Perturb` = current behavior; `FreshZ0` = independent high-variance draw per rollout, seeded, replay-deterministic) behind `eqr_convergence`.
- [ ] T2 — matched-NFE bench (NFE = D·B): FreshZ0+Top1Converged vs Perturb+MostFrequent vs Perturb+BestQ on the `bench_119_eqr_convergence` harness; report accuracy + across-seed variance.
- [ ] T3 — negative-control arm: unshaped fixture set where `Top1Converged` must NOT beat `MostFrequent` (pre-registered before running T2's main grid).
- [ ] T4 — D-first sweep: D ∈ {1, 2, 4, 8, 16, 64} × B grid; publish the measured knee and the law sentence.
- [ ] T5 (optional, observational) — Δ_PI metric + four-mode proxy diagnostics on the bench corpus; feeds the riir-ai deliberation-router composition (riir-ai Issue 881 T5) later.

## Non-goals

- No BoM duplication — `BoMSampler` (Plan 281) stays the single-pass K-hypothesis primitive; a BoM + convergence-selection composition is riir-ai-side (Issue 881).
- No landscape-shaping training (RI/NI training interventions) — that is riir-train Plan 387 Phase 2's spec'd-but-unbuilt territory.
