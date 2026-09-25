# Issue 895: Guided width rollouts — μ≠0 latent guidance, mass-conserving perturbation, decode-free selection (GRAM re-distill)

**Status:** OPEN — filed from Research 590 (arXiv:2605.19376 GRAM), which **SUPERSEDES Research 058** (§7.1 named the μ≠0 gap for the DDTree lane as `SdeConfig.guided`, LOW) and **consumes Plan 095** (GOAT PENDING 1/3 — the width-vs-depth G1/G3 this issue's T7 completes).

## Why

PTRM Plan 083 shipped stochastic width rollouts with **zero-mean** noise (`SdeConfig`, `inject_sde_noise`, `DDTreeBranchCache`, `EarlyStopGate`, `TrajectoryCredit`). GRAM's ablations show the effect is **domain-dependent**: zero-mean 94.88 > guided 93.96 on Sudoku (single-solution — guidance *hurts* there; this is exactly why 058 §7.1 parked it LOW), but guided 99.7 vs zero-mean 50.27 on N-Queens (multi-solution — guidance *essential*). 058's adjudication covered only the **logit-domain DDTree host**; Research 590 routes the same mechanism to the **per-NPC belief host** (`evolve_belief` — deterministic single [f32;8] trajectory today), where three things do not ship anywhere:

1. **Mass-conserving perturbation** — ε drawn in the coexact∪harmonic Hodge subspace, `belief_mass_divergence(ε) ≡ 0` by construction (architecturally stronger than GRAM's own noise; never composed with `hodge_decompose` today).
2. **Decode-free latent selection on a no-reward host** — `BanditPruner` consumes rollout *rewards* (fine in arenas); fog-of-war belief deliberation has no ground-truth reward; self-consistency `Σ_j σ(dot(h_i,h_j))` + convergence residual is the only signal.
3. **Reallocate-to-width** — the shipped trap machinery (`saddle_escape` FlipDetector + `deliberation_trap`/`deliberation_budget`) triggers, kicks, and *caps* budget, but never converts freed budget into parallel hypotheses.

**This issue explicitly OVERTURNS 058 §7.1 (the `SdeConfig.guided`-field-only form) and §8.3's "no new feature flags" — scoped to the belief host** (058 never examined it; a struct field cannot carry cross-domain construction + selection + allocation). 058 §8.3's "do NOT make guided noise the default" **STANDS** — everything here is opt-in/default-off.

## What ships (consume, don't fork)

- `diversity/temp.rs` (katgpt-core) — BLAKE3-seeded deterministic zero-mean noise (the ε source).
- `katgpt-canon` `fit_joint_svd_pair` / `JointSvdFitScratch` — offline joint SVD (the direction-table fitter).
- `katgpt-dec` `hodge_decompose` / `harmonic_projector` / `belief_mass_divergence` — the invariant operators.
- `katgpt-sense` `reconstruction.rs` `evolve_belief` — the belief host.
- `saddle_escape` (`FlipDetector`, `apply_kick`, `TrapObservables`) — the trap-detector + kick substrate (Bench 707 GOAT family; quality-gated PoC posture).
- DDTree rollouts (katgpt-forward/katgpt-speculative) — the token-tree host; **its lane is adjudicated covered (058) — do not add guided noise there**.
- `SigmoidGateCalibrator` / `best_belief_score` — sigmoid scoring discipline.

## Tasks

- [ ] T1 `structured_perturbation` operator (katgpt-core, feature `guided_width_rollouts`, default-off): arm (a) transversal `ε = σ_t · P_⊥ v`, `P_⊥ = I − ûûᵀ` for dense low-d latents ([f32;8] belief — no CellComplex claim, curse-of-dim rule); arm (b) coexact∪harmonic ε for **cochain belief fields** (2D zone maps, d=2): assert `belief_mass_divergence(ε) ≡ 0` by construction.
- [ ] T2 stagnation-gated variance: `σ_t = σ_max · sigmoid(α·(w_stuck − w₀))`.
- [ ] T3 decode-free latent scorer `latent_value` for **no-reward hosts**: self-consistency + convergence residual + optional frozen-direction sigmoid. Where a reward exists, `BanditPruner` stands (058 §5.3) — this operator targets the belief host only. Never publish the score as calibrated probability (if ever consumed as confidence → Report-the-Floor conformal gate).
- [ ] T4 deterministic diversity init: Sobol/BLAKE3-seed branch initialization + farthest-point returned-set selection.
- [ ] T5 success-SVD direction table (the μ≠0 fill, self-adaptive track): outcome-weighted SVD over logged successful Δh (`fit_joint_svd_pair` substrate), per-direction Beta posterior reweight, frozen BLAKE3 table via freeze/thaw; table-absent ⇒ isotropic+transversal fallback bit-identical.
- [ ] T6 trap-kill-reallocate — **consume `saddle_escape`, do not rebuild**: FlipDetector over the belief host's key (the `deliberation_trap` COMPASS_8 / `cgsp_trap_mode` leading-branch precedents); `apply_kick` as the kick; the NEW claims are only (i) belief-state observables (pre-decode), (ii) codifferential-divergence escape signal, (iii) freed budget → width.
- [ ] T7 GOAT gate bench (release profile) — **completes Plan 095's pending G1/G3; cite 095 in the bench header, do not open a third lane**: G1 equal-compute best-of-N×K vs 1×NK on **BOTH families** — a multi-solution/structured family (N-Queens/graph-coloring class, where GRAM's guidance wins) AND a single-solution/uniform family (where GRAM's zero-mean wins, 94.88 vs 93.96) — **pre-stated demote condition: if the direction table (T5) loses or ties on BOTH families, guided stays off-by-default forever (closed-negative, recorded in the bench doc)**. Plus the E9 signature (measured delta over the deterministic arm or the feature is inert) + mass-divergence ≡ 0 arm. G2 O(K) parallel latency; G3 kill-switch bit-identity (`σ=0`/`N=1` ≡ incumbent `evolve_belief` path); G4 zero-alloc (CountingAllocator canary).
- [ ] T8 consumers wired opt-in: katgpt-sense belief host (riir-ai Issue 1008). DDTree host stays out of scope (058's covered verdict).
- [ ] T9 on GOAT PASS + modelless gain → promotion decision per feature-flag discipline (058 §8.3's no-default call governs until a both-families win overturns it — that is T7's demote condition, inverted); record in the per-stack ledger (Research 590 §GOAT).

## Boundary notes

Public substrate only — no game vocabulary (riir-ai Issue 1008 is the consumer). Sigmoid, never softmax. Raw sync paths untouched. T5 delivery via freeze/thaw envelope if the table outgrows katgpt-core's asset discipline.

## Refs

Research 590 (supersedes 058) · Research 058 (prior GRAM verdict — DDTree lane stands) · Plan 095 (width-vs-depth GOAT PENDING 1/3 — T7 completes) · GRAM arXiv:2605.19376 · PTRM arXiv:2605.19943 (Plan 083) · katgpt-rs Research 546 / Plan 593 / Bench 707 (`saddle_escape`) · note 250 (TRM policy improvement).
