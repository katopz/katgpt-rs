# Issue 746 — Looped-Flows modelless extraction candidates (arXiv:2609.11801)

**Status:** OPEN — fusion ideas, novelty TBD (the §1.5 no-candidate-escape-hatch disposition for Path 0 rows that are neither covered, nor training-bound, nor dead). Filed off [riir-train Research 452](../../riir-train/.research/452_Looped_Flows_Training_Recipe_Distill.md) (panel-consolidated distill) + [Plan 395](../../riir-train/.plans/395_looped_flows_recipe_arm.md) (the training arm — owns the consumer-bound sampler rows). This issue holds ONLY the two rows that are modelless-shaped with no consumer inside Plan 395.

## Background (one paragraph)

"Thinking with Looped Flows" trains looped reasoners via local denoising objectives at sorted decreasing noise levels (shared noise, sg-everywhere) and infers by integrating a probability flow coupled to the recurrent state. The distill verdict was GAIN with the modelless track folded into the riir-train harness — the closed-form math is consumer-bound (needs a trained denoiser). The user challenge ("why no modelless?") surfaced that two rows don't fit that disposition: they are closed-form, consumer-shaped, and NOT covered by shipped substrate. They land here, mirroring the R547→Issue 744 pattern.

## Row 1 — Anytime commitment schedule `r_i = Δt/(1−t_i)`

- **Math:** Euler flow update as lerp `x_{t+1} = (1−r)x_t + r·x̂_t` with `r_i = Δt/(1−t_i)`; telescoping law `∏(1−r_j) = (1−t_n)/(1−t_0)` → the initial prior's weight annihilates to EXACTLY 0 at `t_n=1`; on a uniform grid `r = 1/(n−i)` — later steps commit monotonically more.
- **Signal-diff vs shipped:** tf_loop `DampedEuler` (fixed β), RCD residual carry (fixed blend), `CommittedFieldBlend` (sigmoid weights) — none schedule the blend on a time grid; none carry the annihilation guarantee.
- **Candidate consumers:** (a) per-NPC deliberation-within-tick blending (commit progressively, fully forget the initial guess by the end of the window); (b) tf_loop sub-step strategy variant (`FlowEuler`) — only if tf_loop grows a time grid; (c) consolidation blending of a new observation stream into a warm-tier state.
- **Novelty TBD because:** the math is a provable 20-line lerp schedule; the open question is whether any consumer needs *exactly* "monotone commitment + exact prior annihilation" vs a plain sigmoid ramp (which also annihilates asymptotically but not exactly). The exactness is the only edge.

## Row 2 — Marginal-calibrated backtrack (collapse recovery dial)

- **Math:** Eq 18 minus the denoiser: `a = clip[1−γΔt]`, `s = a·t`, `x̄_s = a·x_t + √((1−s)²−(a−s)²)·ε` — rewind a state to an earlier confidence level s and re-noise at the EXACT marginal variance (identity: `a²(1−t)² + (1−s)² − (a−s)² = (1−s)²`). γ is a continuous dial: 0 = no rollback, larger = deeper rewind.
- **Signal-diff vs shipped:** `renoise_ce` perturb-and-remeasure adds UNCALIBRATED noise and measures the CE response; this rewinds to an exact on-manifold level parameterized by confidence. cgsp collapse DETECTION ships; its RECOVERY is restart-shaped. `SpatialBelief` confidence decay `sigmoid(−λ·Δtick)` is a time-parameterized belief — the geometry the formula needs — but lives on scalars, not a simplex interpolant.
- **Candidate consumers:** (a) cgsp collapse recovery — on detection, rewind the latent to level s and re-integrate instead of hard restart (graceful degradation; the paper's γ-SDE is what recovers 98% of spurious attractors); (b) stale-belief re-exploration under fog-of-war (rewind a stale SpatialBelief to re-open the hypothesis space at calibrated variance).
- **Novelty TBD because:** the variance law is specific to Gaussian interpolants on a simplex; our belief latents are not on that manifold. The extraction would be the *schedule* (rewind depth vs staleness/collapse severity), with the variance law adapted to whatever norm the belief state carries. Needs a PoC to see if the calibrated form beats plain noise injection at equal magnitude — if it doesn't, the row dies (defend-wrong discipline).

## Tasks

- [ ] **T1** PoC Row 2 first (higher-value consumer): collapse-recovery bench — calibrated backtrack vs uncalibrated renoise vs hard restart, on a stuck-attractor synthetic (riir-poc or katgpt-core test). Gate: recovery rate at equal noise budget.
- [ ] **T2** If T1 shows no calibrated-form advantage at equal budget → close Row 2 with the measurement recorded; the schedule-only form is not worth a primitive.
- [ ] **T3** Row 1: identify ONE concrete consumer with a time-grid need before writing code (else close — a lerp schedule without a grid is speculative).
- [ ] **T4** On any close: record verdict in Research 452 + remove this issue (house noise rule).

## Cross-references

- Research 452 (Path 0 merge table rows 1–2 context) · Plan 395 T8.2 (dllm-lane secondary mapping — the OTHER modelless residue, consumer-bound) · katgpt-rs R366 pointer · Plan 116 (DiffusionSampler), Plan 136 (tf_loop), Bench 304 (GainCostLoopHalter) — the covered cousins.
