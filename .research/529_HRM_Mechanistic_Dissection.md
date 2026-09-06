# Research 529: Dissecting HRM — Mechanistic Study (Convergence Cadence + Decodable≠Causal)

> Source: [Dissecting Hierarchical Reasoning Models: A Mechanistic Study](https://alphaxiv.org/abs/2609.dissecting-hierarchical-reasoning-models) — Leo Raphael Rodrigues, Jian Kang (MBZUAI), ICML 2026 Mech Interp Workshop (OpenReview `x7a5aCnTzI`, published 2026-06-11; alphaxiv 2026-09-02)
> Local: `.raw/HRM-dissect/` (Apache-2.0, pinned `d956ca85a753ea661bd7b952d4246e197889ed81` — read-only clone, removed after distill)
> Date: 2026, distilled 2026-09-03
> **Verdict: GAIN** — an analysis paper (no new method), but two modelless extractions survive the signal-diff: (1) the **convergence-cadence outcome signal** (‖Δz‖ trajectory separates solve from fail ~4.9× by mid-run → escalation semantics, not just halting), and (2) the **decodable≠causal audit discipline** for our latent-claim surfaces. Hierarchy finding validates two-brain as analyzability dividend.
> **Status:** RECORD — both extractions LANDED (cadence: Issue 720 T1–T4, 2026-09-03→04; causal-audit: riir-ai Issue 858, `fea8c6ef5`). See §Landing record; issue files removed per the noise-reduction rule.

## TL;DR

The paper dissects HRM (27M, the ARC-famous 2-module recurrent Transformer) on Sudoku-Extreme / Maze-Hard / ARC-AGI-2 with causal interventions, probes+directed ablation, and SAEs with size-matched random controls. Four findings:

1. **Iteration is what matters; hierarchy isn't (for accuracy).** Single-state recurrent loops match HRM: vanilla RNN@16 steps **51.4%** ≈ Universal Transformer **48.4%** ≈ HRM **45.4%** (Sudoku, 500 puzzles; measured from the repo's committed per-step evals) while one-pass Plain Transformer = 0.2% and single-step SRNN = 0.0%. Hierarchy's value is *analyzability* — distinct intervention points — not raw accuracy.
2. **State roles are task-dependent.** Sudoku: z_H is the solution repository (ablate/freeze z_H → collapse; z_L ≈ free). Maze: z_L carries the intermediate path (cross-puzzle patching), z_H is the final readout. No fixed division of labor.
3. **Decodable ≠ causal.** Task variables decode from recurrent states at ~90%, but ablating the probe direction (project onto orthogonal complement) ≈ ablating a *random* direction. SAE ablations bite harder than probe ablations, but **top-50 SAE features (-3.63pp) ≈ 50 random features (-3.42pp)** (Table 15) — including under full within-step BPTT. The computation is distributed; there is no compact causally-dominant feature set.
4. **Convergence cadence is the success signature.** Solved runs: ‖Δz_H‖ decays to a fixed point (0.30 at step 7-8), consecutive-state cos → 0.998, state norm GROWS (7.5 → 9.4 — committing confidently). Failed runs: updates plateau HIGH (1.46 — 4.9× the solved value), cos stalls ~0.97, norm stalls ~6.8. Successive updates are near-orthogonal (cos_updates ≈ -0.02..-0.25) — the refinement is largely rotational, not radial.

Overall characterization: **HRM = constraint-aware iterative refinement on a puzzle-specific solution state** — no hierarchy magic, no compact features, but a readable convergence trajectory.

## Why this matters for our stack (three reframes)

**Game/NPC reframe (priority #1).** Our per-NPC belief state is exactly a puzzle-specific solution state refined iteratively (`evolve_belief` leaky integrator, `[f32; 8]`, dot+sigmoid per AGENTS.md bridge rules). Finding 4 is a *free, zero-training* runtime signal for any iterative latent loop: windowed update-magnitude cadence classifies settled vs churning → **settle → commit early; churn → escalate** (deliberation, curiosity, hint-injection, damping). Finding 1 is direct support for the two-brain model: info-brain/think-brain separation is an *analyzability* dividend (intervention points), not an accuracy claim — which is exactly how AGENTS.md frames it ("divergence is emergent, not a bug").

**Healer reframe (priority #2, honest answer: no strong delta).** The healer's fixpoint loop is discrete (propose/no-propose), retrieval is BM25+latent fan-out — no recurrent latent refinement to monitor. Finding 3's methodology (never claim a component matters without a size-matched random-control ablation) is already our house discipline (non-vacuous controls, falsifiable A/Bs, "a gate that cannot fail proves nothing"). Recorded as validation, not action.

**Inference-perf reframe (priority #3).** `forward_looped` (Plan 108, `lt2_looped`) is validated at T=4 only; Issue 717 arms damping/tangential knobs for deep runs. Finding 4 supplies the **trigger** those knobs lack: churn detection tells the runtime *when* to damp (Issue 717's own rule: "don't damp unless inference already degrades" — cadence is the degradation detector). And `GainCostLoopHalter` already consumes step-size ‖Δh‖ for *halting*; the paper adds the missing **outcome semantics**: plateau-high ≠ "stop", plateau-high = "failing — abstain/escalate/restart".

## Signal-diff table (§3.6 — closest shipped cousins)

| Paper mechanism | Closest shipped analog | What the analog consumes | Delta (the gap) |
|---|---|---|---|
| ‖Δz_H‖ cadence → solved-vs-failed | `GainCostLoopHalter` step-size ‖Δh‖ (katgpt-core `gain_cost_halt.rs`, Research 282/Plan 304) | per-loop step size for HALT (concavity decay / expansion growth) | Halting ≠ classification. No outcome read; no escalation arm (abstain/restart/damp); no belief-loop consumer |
| (same) | Issue 717 damping/tangential knobs (lt2 deep loop) | α-swept damping AFTER degradation is observed | Has rescue, no detector — the issue's own rule demands measuring degradation first; cadence IS the detector |
| (same) | `surprise_norm()` / `TemporalDerivativeKernel` (katgpt-sense, Plan 277) | fast−slow EMA derivative of belief → surprise/curiosity | Novelty signal (high churn = interesting), not outcome classifier (high churn = failing); no windowed trajectory, no escalation consumer |
| (same) | `DerivativeCuriosity` (cgsp, Plan 277 F4) | preference-trajectory derivative → explore | Same class: churn → explore, never churn → escalate/abandon |
| (same) | `FreezeGateReport` output_converged/flatness (`can_freeze`, riir-neuron-db) | measure-time consolidation gate | Commit-side only, measured at consolidation, not a live per-tick predictor; validates convergence-gated commitment |
| Probes+directed ablation causal audit | svd_cca Track B (Bench 836: alignment certificate ρ̄) | subspace alignment between adapters/NPC beliefs | Alignment/decodability certificate ≠ causal-use certificate — the paper's exact finding 3; no ablation arm exists in the PoC |
| SAE + size-matched random controls | — (house discipline: non-vacuous controls) | — | No delta; dim-8 NPC beliefs don't need SAE machinery. DISCARDED with reason: control discipline already ships |

## Fusion

**Paper × R35 (Attractor/fixed-point) × Issue 717 (damping) × Plan 304 (halter):** R35's redirect already recorded sotaku's *negative* half (relative-residual is a growing-denominator trap on non-fixed-point recurrences — never a halt signal). This paper supplies the *positive* complement measured on a real solver: the **absolute, windowed** update-magnitude trajectory does carry outcome information (decay → solve; plateau-high → fail), and per finding 1 the informative regime is exactly the near-fixed-point class where R35's IFT line lives. Combined law: (a) absolute Δ, never relative (Issue 717 T6 trap); (b) windowed trajectory, not single-step (plateau vs decay is a shape, not a value); (c) semantics per class — decay→commit/halt, plateau-high→escalate (damp per 717, deliberate per NPC, restart-with-hint per CGSP). Also note cos_updates ≈ 0 supports 717-T4's tangential decomposition: the churn is rotational, so radial damping is the wrong knob when cadence flags failure — scale the tangential component.

## Path 0 decomposition (analysis paper — no training routing)

| Component | Modelless analog | Verdict |
|---|---|---|
| Causal intervention suite (ablate/freeze/patch + random controls) | intervention-style gates exist (engram zero-query test, evidence tripwire rank inversion) | Partial → **LANDED 2026-09-03** — riir-ai Issue 858's directed-ablation arm (`fea8c6ef5`) |
| Linear probes + directed ablation (orthogonal projection) | dot+sigmoid projections are our house style; projection-out is a ~10-line util | No standalone primitive needed; folded into the svcca audit issue |
| SAE + size-matched random ablation | non-vacuous control discipline ships | Covered — discard (dim-8 latents; SAE is high-dim machinery) |
| Convergence cadence outcome signal | NOT covered (see table) | **Gain → katgpt-rs Issue 720 — LANDED (T1–T4), see §Landing record** |
| Hierarchy-as-analyzability principle | two-brain model (info/think) | Validates; doc-level, no code |

Training track: the paper trains probes/SAEs/one BPTT control for *analysis only* — no recipe, no optimizer/loss content. Path 0.5 not applicable; adversarial panel skipped (classification is interpretability-analysis; the abstract carries probing/SAE framing, not optimizer/backprop framing, and no riir-train routing decision exists to adversarially test).

## Verdict scoring (§1.5)

- Q1 prior art: in-stack — no duplicate (table above). Published — convergence-residual early exit is standard in fixed-point/DEQ/ACT literature, and the paper itself + arXiv:2601.10679 ("Are Your Reasoning Models Reasoning or Guessing?") + arXiv:2605.20784 (Interaction Locality) own the HRM-analysis landscape. Our claim is an **application fusion** (cadence → escalation semantics on NPC belief loops + looped-inference triggers), not a new mechanism → not Super-GOAT.
- Q2 new behavior class: escalation-on-churn (abstain/deliberate/damp/restart) is a capability our halter/curiosity pair lacks — but it is a recombination of shipped ingredients → GOAT-tier at best.
- Q3 selling point: "our NPCs detect their own indecision from belief churn and deliberately escalate; settled beliefs commit early — zero training" — finishable.
- Q4 force multiplier: connects halter + 717 + surprise kernel + deliberation + can_freeze (≥2 pillars) — yes.

**Not all four YES at the strength Super-GOAT requires (Q1 fails at mechanism level) → GAIN.** Files: katgpt-rs `.issues/720` (cadence gate), riir-ai `.issues/858` (svcca causal-use audit).

## Key numbers (from the repo's committed aggregates)

| Signal (Sudoku, z_H) | Solved (n=93) | Failed (n=107) |
|---|---|---|
| ‖Δz_H‖ @ step 7-8 | 0.30 | 1.46 (~4.9×) |
| ‖z_H‖ trajectory | 7.5 → 9.4 (grows) | stalls ~6.8 |
| cos(z_t, z_{t-1}) | → 0.998 | stalls ~0.97 |
| cos(Δ_t, Δ_{t-1}) | ≈ 0 (orthogonal steps) | ≈ 0 |

Baselines (500 Sudoku puzzles, final step): RNN@16 51.4% / UT 48.4% / HRM 45.4% / Plain-Transformer@16 0.2-0.4% / single-step 0.0%. Cross-task: top-50 SAE -3.63pp vs random-50 -3.42pp. Maze: z_L carries the intermediate path (patching), z_H the readout.

## Landing record (2026-09-03 → 2026-09-04; both extractions landed, issue files removed)

### Extraction 1 — convergence-cadence outcome signal (katgpt-rs Issue 720, T1–T4 DONE)

- **T1 — `ConvergenceCadence` probe** (katgpt-core, opt-in `cadence_gate`): zero-alloc ring of last-K update norms, caller-fed, emits `Settled { mag } | Churning { mag, plateau_len }` from decay-ratio + plateau detection. G1 bit-identical when off; G4 zero-alloc hot path; shuffled non-vacuity. Landed `99920de2` (12/12 feature-on tests incl. paper-shape fixtures; default 1992/0 bit-unchanged).
- **T2 — falsifiable A/B** (riir-poc `cadence_gate_poc`, `b18b7b2bb`): three arms on a controlled d=16 loop, three dynamics families (contraction-solvable / rotational-churn / reversal-churn). Verdict (all PASS):

  | gate | result |
  |---|---|
  | G1 determinism | all 5 arms bit-identical double runs |
  | G2 compute + no-regression | halt-only 2048 vs cadence 1044 steps on rotational churn (**1.96×**); solvable+reversal bit-identical |
  | G3 abort precision/recall | **1.000 / 1.000** (32/32 rotational flagged, 0 false aborts) |
  | G4 shuffled non-vacuity | control fires 46 flags, precision 1.000 → 0.696, gap 0.304 — the probe reads windowed SHAPE, not magnitude |
  | G5 damp honest-negative | 0 compute saved, 0 flags, err 5.75 → 4.64 (19% radius shrink, no recovery) |

  En-route findings pinned in the harness header: single-plane rotation = degenerate world (14-dim fixed-point axis; sub-floor plateaus invisible to BOTH signals — caught by the pre-registered gates, fixed to the 8-plane direct sum); **θ=90°: the halter's oscillation detector fires on f32 noise at true cos 0** — a real `GainCostLoopHalter` boundary characteristic (suite keeps cos θ ≥ 0.259).
- **T3 — NPC consumer** (substrate riir-ai `681786288` + SDK forward `69a0770` + riir-mmorpg-examples `8c99624`): heading-churn windowed trigger (catches gapped oscillators the raw `flee_ticks` counter misses) + settled early-commit + corner preservation behind the opt-in `deliberation_cadence` feature; system-owned shadow state, think-brain only. G1–G4 + signal-vs-counter non-vacuity ALL PASS at 1000 NPCs (debug + release); CI gate coverage Layer 1.9b. The consumer-side doc record lives in riir-mmorpg-examples HISTORY.md (Issue 054, "L2 trigger upgrade"). **Promotion to default is the owner-gated gameplay A/B — the reopen trigger.**
- **T4 — doc pins**: halter "Halt ≠ classification" block `4cff9830`; riir-neuron-db `can_freeze` live-prediction-sibling note `9e45d42`.
- **T5 `- [-]` deferred**: CGSP restart-with-new-conjecture arm (DerivativeCuriosity owns the explore axis; add the abandon axis only if a consumer shows explore-alone is insufficient).

### Extraction 2 — decodable ≠ causal (riir-ai Issue 858, RESOLVED `fea8c6ef5`)

The svd_cca PoC gained the directed-ablation causality matrix (`g4_issue858_directed_ablation_causality_matrix`): carrier direction = top left singular vector; ablation `z ← z − (zᵀẑ)ẑ` with a DISTINCT seeded readout (the certificate must not grade its own ablation); 16-draw random control. Measured fixture cell = **aligned+causal** (Δ 1.2327 vs band [0.0287..1.0640]); ρ̄ itself is ablation-invariant (0.9998 → 0.9998) — alignment spread across directions, not concentrated in the carrier. "Alignment ≠ causation" paragraph + required-evidence rule pinned in riir-ai Bench 836 §Disposition. T4 (reusable `causal_probe` util) deferred until a second claimed-readout consumer materializes.

## References

- HRM corpus line: R9 (HRM), R10 (TRM), R11 (Sotaku), R35 (Attractor/fixed-point + Sotaku DEQ redirect), R48 (HRM-Text), R50 (LDT), R58 (GRAM)
- Issue 717 (lt2 damping/tangential/f32-state + residual trap) — the runtime-rescue cousin; this note supplies its missing detector
- Plan 304 / Research 282 (GainCostLoopHalter — step-size halt semantics), Plan 277 (surprise kernel + DerivativeCuriosity)
- riir-neuron-db `can_freeze` (convergence-gated consolidation — validated), riir-ai Bench 836 (svcca Track B — audit target), riir-mmorpg-examples Issue 054 (L2 deliberation — churn-trigger consumer)
- Prior art named by the paper: arXiv:2601.10679 (Reasoning or Guessing — NOT yet in our corpus; closest external cousin), arXiv:2605.20784 (Interaction Locality)

> **PASS-Redirects (synthesis):** Rodrigues & Kang [alphaxiv 2609.dissecting-hierarchical-reasoning-models "Dissecting Hierarchical Reasoning Models: A Mechanistic Study"] — GAIN verdict (issue 720 + riir-ai 858); convergence-cadence outcome signal is the positive complement to this note's IFT/Sotaku negative half, and the decodable≠causal finding is the audit discipline for any latent-claim surface.
