# Research 546: Fractal Basins Trap Latent Reasoning — Saddle-Trap Escape for Looped Latent Reasoners

> **Source:** [Fractal basins trap latent reasoning](https://arxiv.org/abs/2609.04963) — Jeffrey Lai, Anthony Bao, John Quinn, William Gilpin (UT Austin: Oden Institute + Dept of Physics), 4 Sep 2026, arXiv:2609.04963v1, 6 pp + appendices. Basin-probe code: github.com/GilpinLab/loopscape.
> **Date:** 2026-09-11
> **Status:** Active — **Super-GOAT (scoped to the gated three-way controller)**. Filed this session: katgpt-rs Plan 593 (open primitive `saddle_escape`, opt-in) · riir-ai Research 374 (game-runtime guide) · riir-train Issue 532 (training diagnostic) · riir-clippy Issue 91 (healer fusion idea). Quality claims UNPROVEN until Plan 593 Phase-3 PoC (§3.6 gating).
> **Related Research:** 079 (EqR — a studied model; its "zero ROI" Lyapunov rejection challenged below) · 266/466 (FPRM — a studied model; its residual-τ halting is the paper's specimen) · 282/304 (gain/cost halting — the integration surface) · 525 (structural CoT halt — decode-keyed answer ring) · 270 (ICT collision purity / BranchingDetector) · 406 (renoise-CE perturbation-drift probes) · 371 (mean-field saddle detection — crowd order-parameter level) · 344 (implicit-FP halting survey) · 317 (reasoning as attractor dynamics) · 536 (EBT) · 243 (Bebop forecast) · 218 (breakeven router) · 350 (density-aware scheduling) · 125 (MCTS collapse discriminator)
> **Related Plans:** 304 (`GainCostLoopHalter`), 294 (ICT detector), 085 (`ResidualRelevanceScorer`), 108/136 (`LoopMode::*`), 276 (micro-belief), 231 (PathwayTracker)
> **Classification:** Public (gate kernel → katgpt-rs); NPC wiring selling point private (riir-ai 374).

**Pinned claim (§1.5, written before the prior-art searches):** a **chaos-indicator-gated, deterministic, three-way loop controller** — Continue / Halt{Converged} / Halt{Trapped} / Kick — for latent/looped reasoners, consuming only per-loop trajectory observables (oscillation streak + decode-flip-rate EMA + optional one-step probe drift), distinguished from the shipped halter family by the **Trapped halt semantics** and the **conditional perturb-and-resume kick**, and from published perturb-and-resume (PTRM arXiv:2605.19943) by being **trap-conditional, deterministic, single-trajectory** rather than always-on stochastic noise over parallel rollouts + voting.

## TL;DR

**Verdict: Super-GOAT (scoped).** The paper is a diagnosis: looped latent reasoners (EqR/HRM, FPRM, Parcae, TRM) are dynamical systems whose initial-latent-space basin boundaries are **fractal**, fractality (basin entropy) tracking task difficulty; reasoning slowdowns are **transient chaos** — trajectories scattering off weakly-unstable **saddles that decode to nearly-correct answers** (maze dead ends, Sudoku repeated-digit grids). The operational extractables are cheap: λF via neighbor-trajectory divergence; **solution-switch frequency** (decode every loop, count answer changes) correlates with λF; basin entropy correlates with loop count.

Our shipped halting family is literally the paper's specimen set (FPRM's residual-τ halt = `ResidualRelevanceScorer`, Plan 085; EqR = Research 079; looped blocks = `LoopMode::WeightShared`, Plan 108) — and every halter in the stack is **two-way** (Continue/Halt). The paper's mechanics imply a **three-way** controller the stack lacks: halt-on-oscillation (Plan 304) *gives up* holding a nearly-correct answer, while fractal-basin structure says a bounded deterministic kick can reroute a trapped trajectory onto a different route. Novelty (searched, §3): bare perturb-and-resume is folk practice (PTRM always-on noise; GTS), but **trap-conditional + deterministic + single-trajectory gating is unclaimed**, **trapped-vs-converged halt semantics is novel**, and **decode-per-loop flip-rate budget allocation in a latent loop is unclaimed** (ES-CoT/TrACE are token/rollout-space). Quality claims are PoC-gated (Plan 593 Phase 3) — unproven until run.

## 1. Paper core (what it proves)

1. **Probe:** hold model+prompt fixed, vary only the initial latent state along a random 2-D slice (QR of a Gaussian), 200×200 grid; **settling time** = loops until the decoded output stops changing → the settling-time field is the basin map.
2. **Fractality tracks difficulty:** basin entropy (Daza et al. 2016) correlates with mean convergence loops across Sudoku-Extreme / Maze-Hard / Countdown / ARC-AGI and across all four architectures. Infinitesimal init changes ⇒ orders-of-magnitude convergence-time changes.
3. **Mechanism — transient chaos:** no sustained chaos (all trajectories converge); weakly-unstable **saddles scattered through phase space** redirect trajectories for extended durations ("Plinko" scattering). Escape time grows with the number of unstable directions.
4. **Saddles = nearly-correct answers:** decoding latent states near saddles yields almost-solutions. λF (fast Lyapunov indicator — max decoded-state deviation from spatial neighbors over time) marks basin boundaries = maximally-uncertain routes; λF **correlates with the number of solution switches** per trace (decode-per-loop flip rate).
5. **Training bifurcation:** a looped transformer trained on `Ax=b` over F_p bifurcates at ~150k steps: incorrect solutions lose stability (become saddles), basin entropy jumps, unstable Jacobian eigenvalues appear — and **FTLE is positive only on "core" variables** (multi-step Gaussian elimination), never on leaf variables (direct substitution). Transient chaos **is** the multi-step reasoning algorithm; flattening basins would delete the capability.
6. **Slim fractals (App. E):** some models show scale-dependent fractal dimension — "doubly transient chaos": the fixed-answer convergence bias acts as dissipation decaying the saddle set.

Context: overthinking nearly doubles inference cost and adversarial prompts induce 10× compute DoS (their ref [8], arXiv:2602.00154) — an online trap/hardness signal is a serving-plane security surface, not only a quality one.

## 2. Vocabulary crosswalk (grep both sets)

| Paper term | Codebase equivalents | Where it ships |
|---|---|---|
| transient chaos / long transient | sustained direction-reversal oscillation; "update magnitude expanding, not contracting" (concavity floor) | `GainCostLoopHalter` `HaltReason::{Oscillation, Concavity}` (Plan 304 + Issue 698 T4) |
| saddle = nearly-correct solution | decoded answer flipping between candidates; answer-ring revisit | `structural_cot_halt` ring + SelfLoop/BacktrackRevisit (Plan 525); SweTrajectoryFreezer "committed_wrong" class |
| fast Lyapunov indicator λF | JS-divergence uniqueness vs population; one-step perturbation drift | `ict::BranchingDetector` (Plan 294); `renoise_ce` probes (Bench 406 double-well: saddle-near = high drift) |
| basin / settling time | failure-mode trajectory shapes (oscillation / committed-wrong / converged) | SweTrajectoryFreezer (Benches 012/013/015); micro-belief flip-flop counts (Bench 276) |
| solution-switch frequency | decoded-key flip rate per loop | **NEW** — the ring exists (525); nobody consumes flip *rate* |
| escape time ∝ difficulty | Bebop acceptance forecast; breakeven router; density scheduler | Plans 243/218/350 — all forecast cost/acceptance, none forecast loop demand from flip rate |
| saddle escape (perturb + resume) | kick + resume looping | **NEW** — nothing perturbs a *live* loop's state and resumes (renoise-CE kicks *candidates* for selection, not a running trajectory) |
| crowd saddle (order-parameter level) | `saddle_strength` / `saddle_margin` (2×2 Jacobian eigenvalues) | `mean_field_regime` (Plan 371, DEFAULT-ON) — crowd scale, not per-trajectory |

## 3. Published prior art (skill §4 — subagent, ~20 searches)

| Work | Does | Delta vs scoped claim |
|---|---|---|
| **PTRM arXiv:2605.19943** (May 2026, TRM family) | **Always-on Gaussian noise per recursion step** + Q-head selection; names the "no escape mechanism" problem | Closest threat to the kick. Always-on stochastic over parallel trajectories + voting — not trap-conditional, not deterministic, not single-trajectory, no three-way gate |
| GTS arXiv:2602.14077 | Learnable Gaussian thought sampler (GRPO-trained); documents heuristic latent perturbation as the standard ITS baseline | Needs training; confirms perturb-and-resume per se cannot carry novelty |
| SoftCoT++ arXiv:2505.11484 | Perturb via multiple initial tokens | Init-time diversity, not mid-loop escape |
| STARS arXiv:2605.26733 (ICML 2026) | Loop-collapse problem; fix = training-time Jacobian spectral-radius regularization | Problem known; their fix is training-side — the inference-time modelless alternative is open |
| ES-CoT arXiv:2509.14004 | Consecutive-identical-answer run-length → halt | Flip-signal family, token space, halt-only, no budget allocation |
| Answer-convergence arXiv:2506.02536 / TrACE arXiv:2604.08369 | Answer-consistency halting / rollout agreement → per-step budget | Token/rollout space; decode-per-loop latent budget allocation unclaimed |
| FPRM arXiv:2606.18206 / RD-VLA arXiv:2602.07845 | Residual/latent convergence halting | Two-way Converged/Halt; no trap class, no kick |
| Storm et al. PRL 132, 057301 (arXiv:2306.12548) | FTLE ridges in feed-forward input space | Diagnostic; no loop, no control |
| Chaos-in-reason arXiv:2607.27805 | Lyapunov analysis of CoT token dynamics | Analysis only |
| Stop-Overthinking survey arXiv:2503.16419; ROM 2603.22016; TACT 2605.05980 | Taxonomy has **no perturb-and-resume category**; ROM = trained boundary-forcing; TACT = steering toward calibrated directions (trained probe, corrective) | The overthinking line is uniformly exit/compress/steer — never detect-trap-and-kick |

**Search verdict (subagent):** (a) kick partially covered — bare mechanism published, **gated deterministic trap-triggered kick does not exist**; (b) trapped-vs-converged halt semantics **novel**; (c) flip-rate budget partially covered (token space only). The integrated three-way gate with a chaos-indicator trap detector is the defensible novel core; claims scoped to the gating/detection layer. Must-cite trio: PTRM + STARS + ES-CoT.

## 4. In-stack prior art + signal-diffs (each "covered" cousin diffed per §3.6)

| Shipped | Signal it consumes | Diff vs the paper's mechanism |
|---|---|---|
| `GainCostLoopHalter` (304) | gain vs cost curves; cos θ reversal streak; step growth (concavity) | Two-way: Continue/Halt. Oscillation ⇒ **give up holding a nearly-correct answer**; no trap semantics; no resume |
| `structural_cot_halt` (525) | decoded-answer ring (self-loops, revisits) | Halts on structure; never perturbs/resumes; flip **rate** unconsumed |
| `ict::BranchingDetector` (294) | JS divergence of K trajectories to population mean | λF-family statistic exists — consumed for branching decisions, not trap/budget; rollout populations, not a live loop |
| `renoise_ce` (406 + Bench 406) | one-step perturbation drift per candidate | Selection-time scoring of static candidates; never re-enters a running loop (signal-diff: candidate-level, not trajectory-control) |
| SweTrajectoryFreezer (012/013/015) | trajectory shape (π-curvature) | Classifies oscillation/committed-wrong/converged; not wired into any halter; no escape action |
| micro-belief flip-flops (276) | argmax flips per tick | Evaluation metric for kernels, not a runtime gate |
| `mean_field_regime` saddles (371) | 2×2 crowd Jacobian eigenvalues | Crowd order-parameter scale; not per-trajectory latent loops |
| `ResidualRelevanceScorer` (085) | relative residual | Exactly FPRM's halt test — the paper shows residual-τ halting is blind to *why* convergence is slow |
| Research 079 (EqR) | — | Rejected "Lyapunov exponents / basin volumes" as zero-ROI. The paper does not rehabilitate full spectra — but the O(1) proxies (neighbor-divergence λF, decode-per-loop flip rate) are operationally valuable (addendum added to 079) |

**Reverse-grep for documented gaps:** the halter's own GRT notes (Issue 698 T4) document "extending R past the trained R degrades" — a known failure shape whose only mitigation is halt. The healer has no fix-loop trap detector (riir-clippy Issue 91). Gaps confirmed from code comments and bench ledgers, not invented.

## 5. Novelty gate (Q1–Q4, scoped claim)

| Q | Answer | Evidence |
|---|---|---|
| **Q1 no prior art** | **YES** (scoped) | §3: gated/deterministic/trap-conditional kick, trapped-vs-converged semantics, decode-per-loop latent budget control — none published. §4: none shipped — kick, trap reason, and online flip-rate→budget all absent; every near cousin signal-diffed above |
| **Q2 new behavior class** | **YES** | Two-way → three-way control: active conditional escape + halt outcomes that carry state (Converged vs Trapped ⇒ different downstream handling). A capability the family lacks, not better numbers |
| **Q3 product selling point** | **YES** | "Our latent reasoners and NPCs detect when they are circling a nearly-correct answer and take one bounded deterministic escape kick instead of burning budget or returning a near-miss; the same flip-rate signal caps adversarial overthinking on the serving path." |
| **Q4 force multiplier** | **YES** | Composes ≥4 shipped surfaces: halter (304) + answer ring (525) + population divergence (294) + probe drift (406); downstream: think budget (339), deliberation cadence, MCGS collapse modes (125), serving DoS guard |

**All 4 YES → Super-GOAT.** Mandatory outputs filed this session (no "candidate" hatch): Plan 593 (open primitive), riir-ai Research 374 (guide), riir-train Issue 532, riir-clippy Issue 91.

## 6. Latent reframe + boundary compliance

- Everything is **latent-to-latent**: the gate consumes trajectory observables (step norms, cos θ, decoded-key hashes, probe drift) — never decoded text; the kick perturbs the latent state by ε·u, u a BLAKE3-seeded unit vector (deterministic, bit-reproducible), ε annealed (0.5^k), kick budget bounded (default 2).
- **Two-brain compliance:** the kick perturbs the **think brain only** (per-NPC belief / latent loop state). `MapPos`/physics/synced raw state untouched; nothing new crosses `SyncBlock`. If surfaced at all, Trapped/Converged exit as scalars (flip-rate EMA, kicks-used) — bridge-compatible.
- Zero-alloc decision path, fixed scratch, sigmoid-gated thresholds — same contract as the halter family.

## 7. Game-context + consumer-context reframes (mandatory)

**Game (priority #1):** NPC deliberation loops are exactly trapped-dynamics territory: a think loop flipping between two nearly-right plans (attack vs retreat; which apple; which quest) is the paper's saddle-circling, visible as decode-key flip rate. Shipped today: `KnpcSelector` (wraps the halter for planning horizon), `deliberation_cadence` (heuristic stuck-NPC trigger), think budget (Research 339), MCGS collapse discriminator (125). The trap gate upgrades all four: principled stuck trigger; kick = inject bounded noise into belief → try the other plan (emergent "reconsideration"); Trapped → downshift cadence instead of near-miss commitment; crowd-level flip rate = zone indecisiveness scalar (latent, local, never synced). Full wiring map: riir-ai Research 374.

**Consumer (healer, priority #2):** `--fix-compile` iterates fix→compile cycles; an oscillating fix pair (error A → fix B → error B → fix A) is saddle-circling between near-fixes. Flip rate over attempted fixes per span → escalate/switch domain/decline instead of burning fix budget (L4 datum: ~191 s/fix, 0/60 EM — trapped loops are expensive here). Filed PoC-first: riir-clippy Issue 91 (novelty TBD — mine `.heal/` trajectories for prevalence before building anything).

## 8. Three-track verdicts (one per track) + audited scope cut

- **Modelless (PRIMARY — serving-envelope fit): the three-way gate.** Runs inside the loop hot path (per-loop O(1) decision). Filed: Plan 593.
- **Model-based (SECONDARY): the training diagnostic.** Advocate verdict (web-verified): the bifurcation finding supports a **cheap K-rollout FTLE/λF + flip-rate probe** logged in eval loops of looped solvers (Sotaku/LT2/ELT) — capability attribution (core-vs-leaf) and checkpoint selection (equal accuracy, lower settle time) that accuracy curves cannot provide. Honest caveats: paper evidence n=1 (basin-entropy jump is coincident with the accuracy jump, not demonstrably leading); chaos-scheduled curricula are unpublished white space (record, don't build); strong-form "train basins less fractal" is argued against by the paper itself (fractality = the capability). **Not Bonsai** — single-pass drafters have no latent iteration to probe. Filed: riir-train Issue 532 (PoC-first).
- **Audited scope cut (mechanism-level, per §3.5):** "bare perturb-and-resume" is NOT claimed as novel — PTRM 2605.19943 ships always-on stochastic noise in the same model family and GTS documents it as the ITS baseline. The cut cites the uncovered delta: conditioning (trap-detector-gated), determinism (BLAKE3-seeded, bit-reproducible), single-trajectory control (vs parallel noise + voting), and the Trapped halt semantics PTRM names but never detects. No other advocate finding discarded.

## 9. Fusion (closest cousins × this paper)

- **304 halter × 525 answer ring × paper flip-rate → the trap detector.** Oscillation streak alone can't separate saddle-circling from productive backtracking; flip-rate EMA over decoded keys adds the "switching between nearly-correct answers?" axis; the paper's Fig 4B correlation is the empirical license.
- **406 renoise-CE × live loop → the kick.** Reuses perturb-and-measure at trajectory level: probe drift confirms trap; kick + resume attempts escape; budget exhaustion ⇒ Trapped.
- **294 BranchingDetector × crowd → λF proxy at zone scale.** Per-rollout JS uniqueness ≈ fast Lyapunov indicator; aggregated = zone indecisiveness scalar for crowd dynamics (guide §Wiring).
- What none has alone: **bounded-latency escape from nearly-correct entrapment with honest halt semantics** — halters stop, selectors reschedule, probes score; nothing escapes.

## 10. MOAT gate

`katgpt-rs` scope: looped/latent inference primitive — squarely in the transformer-stack/halting lane. Feature flag `saddle_escape` (opt-in); GOAT gates + PoC before any promotion (mirrors the 525/Bench-834 precedent). Demotion check: no primitive loses a slot — this **extends** the halter by composition (no fork; `HaltReason` untouched; gate-level enum added). `riir-ai` MOAT: pillar-level (reasoning × self-adaptive runtime); NPC wiring private. Public/private split per commercial strategy: kernel public in katgpt-core; game selling point private in Research 374.

## 11. Plan of record + PoC status

- **katgpt-rs Plan 593** — `crates/katgpt-core/src/saddle_escape.rs`, feature `saddle_escape`: TrapDetector (oscillation-streak reuse + flip-rate EMA over caller-supplied decoded keys + optional probe drift), `SaddleEscapeGate` (Continue/Halt{Converged}/Halt{Trapped}/Kick), deterministic BLAKE3-seeded kick, Phase-3 defend-wrong PoC (gate vs halt-only vs always-on-noise on a double-well-with-saddle + hard-CSP toy), Phase-4 GOAT gates (≤500 ns/decision, 0 allocs, flag-off bit-identical, bit-repro).
- **riir-ai Research 374** — game-runtime wiring guide (KnpcSelector, deliberation_cadence, think budget, MCGS, crowd λF, DoS guard).
- **riir-train Issue 532** — training-diagnostic PoC (K-probe FTLE/flip-rate; looped solvers only).
- **riir-clippy Issue 91** — fix-loop trap detector fusion idea (PoC-first).
- **Research 079 addendum** — operational exception to the Lyapunov rejection.

**§3.6 PoC status: NOT YET RUN.** Architectural claims (§4 greps + §3 searches) stand on grep/read evidence; the quality claim — "kick escapes traps and improves solve rate vs halt-only without accuracy regression" — is explicitly **unproven** until Plan 593 Phase 3 passes. No promotion before that.
