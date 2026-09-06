# Research 536: Energy-Based Transformers — Scalable Learners and Thinkers (EBT)

> **Source:** Gladstone, Nanduru, Islam, Han, Ha, Chadha, Du, Ji, Li, Iqbal — [arXiv:2507.02092](https://arxiv.org/abs/2507.02092) "Energy-Based Transformers are Scalable Learners and Thinkers" (UVA/UIUC/Stanford/Harvard, Jul 2025; ICLR 2026 per OpenReview). Code: github.com/alexiglad/EBT.
> **Date:** 2026-09-06
> **Status:** RECORD — training track filed → `riir-train/.plans/388_ebt_energy_head_dspark_calibration.md`; modelless track **Pass** (covered; deltas audited-discarded below).
> **Related Research:** 346 (**IRED — the direct predecessor, this paper's own ref [67]**), 317 (Reasoning as Attractor — Gibbs energy scoring), 316 (DSpark confidence-scheduled decoding — the consumer of the trained head), 243 (Bebop entropy-bounded MTP acceptance), 266 (FPRM damped fixed-point halting), 344 (implicit fixed-point RNN halting), 525 (TRACE overthinking halting), 350 (density-aware compute scheduling), 100 (EGA energy-gated attention), 466 (Hopfield top-eigenvector recall), 466/317 energy-vocabulary precedents.
> **Related Plans:** riir-train 388 (filed this session), katgpt-rs 575 (RiskControlledExit — the calibrated-exit substrate), riir-train 342 (GenRM L4 verifier — the generative sibling of the discriminative recipe), 243/294 (Bebop acceptance calibration).
> **Classification:** Public — generic EBM/thinking primitives; no game/chain/shard semantics in the modelless half. Training recipe → riir-train.
> **PASS-Redirects (synthesis):** Huang et al. [arXiv:2405.19715 "SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths"] — PUBLISHED prior art for the trained-acceptance-head class (train a head predicting conditional acceptance probability; adapt candidate length). This is why riir-train Plan 388 is engineering (fixing the measured Bench-696 calibration defect on OUR serving path), not a novelty claim. Also Liu et al. [ACL 2025 "A Drop-In Solution for On-the-Fly Adaptation of Speculative Decoding"] — on-the-fly head adaptation; recorded for the plan's Phase-3 reading list.

---

## TL;DR

EBT trains an energy function `E_θ(x, ŷ)` over (context, candidate-prediction) pairs and **predicts by gradient descent on the prediction itself** (`ŷ ← ŷ − α∇E`), backpropagating through the whole optimization (Hessian-vector products; ≈1.66× a forward per step, ≈3.33× a standard training step, 6.66× total at 2 steps). Three claims matter to us:

1. **Scaling:** first architecture to out-scale Transformer++ pretraining (up to +35% scaling rate on data/batch/params/FLOPs/depth) without a tokenizer change — across text AND video.
2. **Thinking:** inference-time gains from (a) more optimization steps and (b) BoN + min-energy self-verification — +29% more than Transformer++ on OOD language tasks; image denoising beats DiT with **1% of the forward passes**. Gains grow **linearly with OOD-ness**.
3. **The energy scalar is a learned uncertainty signal**: low energy on predictable tokens ("the"), high + non-convergent on hard tokens; high energy on OOD sequences ("knows what it doesn't know") — epistemic uncertainty for free.

**Routing:** the substantive half (the trained energy head) is **training** — R346 already proved the three modelless-unblock paths fail for this exact class (the energy is a *learned* constraint function; deterministic energies degenerate to smoothing). The modelless half (signal shapes: plateau-stopping, BoN-min-selection, compute-allocation laws) **ships** under different vocabulary — audited row by row below. The one actionable seam is the training track: EBT's discriminative-energy recipe applied to the **measured DSpark confidence-head overconfidence defect** (Bench 696: reported 0.75–0.79 vs 0.288 measured acceptance) → riir-train Plan 388.

---

## 1. Paper Core Findings

### 1.1 Method (Algorithm 1/2)

- Train `E_θ(x, ŷ)` by **optimization-training**: initialize `ŷ₀ ~ N(0, I)`, run N gradient-descent steps on ŷ, apply the task loss (CCE / MSE / Smooth-L1) **only at the last step**, backprop through the entire chain via HVPs. This pushes the landscape locally convex around ground truth — the regularizer that replaces contrastive negative sampling (which dies of curse-of-dimensionality).
- **Energy landscape regularization (ablated, Table 2)**: replay buffer (longer simulated trajectories), Langevin noise on predictions (landscape exploration), **randomized step size α** (per-sequence-element — per-batch destabilizes) and **randomized number of steps (2–3)**. Full config: +18.7% thinking; removing randomized α nearly kills it (−1.47% thinking-longer).
- **S1 vs S2 curriculum**: S1 = detach predictions between steps + loss every step + learnable α (stable, no thinking). S2 = no detach + loss-at-last-step + truncated backprop + fixed α + all regularization (thinking emerges). S2 scales at the same-or-better rate with a higher intercept. Tuning order: S1 first, migrate gradually.
- **Architecture**: decoder-only EBT makes all N predictions in parallel — observed block `z_o` + predicted block `z_p` (tensors B×2N×D), superdiagonal per-prediction self-attention trick (~2× FLOPs, not 4×); step embedding (learnable per-optimization-step embedding) critical for stability; Llama2 recipe (RMSNorm/Xavier/SwiGLU/RoPE); input-distribution normalization (softmax before embedding) critical. Bidirectional variant for masked/denoising.

### 1.2 Results

| Axis | Result |
|---|---|
| Pretraining scaling | up to **+35%** rate vs Transformer++ on 6 axes (data/batch/depth/params/FLOPs/width); confirmed on FineWeb at larger scale |
| Thinking (language) | **+29%** more improvement than Transformer++ from extra forward passes; Transformer++ gains exactly 0 per-token |
| Self-verification scaling | BoN-5 benefit grows 4–8% → 10–14% as data grows 10× (thinking capability itself scales) |
| Adversarial dynamics | small models find low-energy-but-wrong candidates (BoN-10 < BoN-2); **shrinks with data scale** |
| OOD law | thinking gains grow ~**linearly with OOD magnitude** (downstream/pretraining perplexity ratio) |
| Generalization | EBT beats Transformer++ on most downstream tasks **despite worse pretraining perplexity** (verification generalizes better than amortized generation) |
| Uncertainty | energy correlates with token predictability (Fig 8) and scene predictability (Fig 11); high energy on OOD sequences (Fig B.2, epistemic) |
| Image denoising | beats DiT in/out-of-dist with **99% fewer forward passes**; linear-probe ImageNet ~10× DiT accuracy |
| Limitations | 3.33–6.66× training FLOPs; α sensitivity; **multi-modal distributions fail** (convex-landscape training merges modes → blur; text-to-image generates blurry averages); tested only to 800M params / 10²¹ FLOPs |

### 1.3 Positioning vs the stack's prior art

**vs R346 (IRED, arXiv:2406.11179 — this paper's ref [67]):** IRED proved the *method* on small structured tasks (Sudoku 99.4%, matrix inverse) with task-specific MLPs and **avoided** backprop-through-optimization (denoising + contrastive losses instead). EBT is the scaled successor: transformer architecture (AR + bidirectional), optimization-training WITH backprop-through-optimization, the regularization ablations, the S1/S2 curriculum, and — the new empirical content — **the scaling evidence** (first to out-scale Transformer++) and **the uncertainty/OOD laws**. R346's verdict mechanics carry over unchanged: the energy head is a learned function; paths 1–3 fail. What R346 could not have said: the recipe now has scaling evidence and a stability curriculum behind it.

**vs R316 (DSpark):** DSpark's confidence head is exactly a trained per-position acceptance-energy head — and our serving of it (Bench 696 G2-full) measured it **overconfident** (0.75–0.79 reported vs 0.288 TRUE top-1 acceptance; quantized-target + domain shift). EBT's discriminative framing + SpecDec++'s acceptance-head prior art = the fix recipe → Plan 388.

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase)

| Paper term | Codebase equivalent (shipped) | Location |
|---|---|---|
| Energy scalar as uncertainty | `contrastive_scope` D-statistic (input-side); fusion margin (max−second goal score, discrete candidates); `RiskControlledExit` confidence trajectories | katgpt-core `scope_model`/`contrastive_scope`; riir-games motivation fusion; katgpt-core Plan 575 |
| Convergence-based stopping | `gain_cost_halt` (FPRM, R266); implicit fixed-point halting (R344); TRACE structural halting (R525); deliberation settled-early-commit | `gain_cost_halt`; riir-games `SwarmDeliberationSystem` |
| BoN + min-score self-verification | `BoMSampler` (Plan 281); `boltzmann_probabilities`; MCTS leaf eval; `AcceptanceSurrogate::expected_accepted_length` | katgpt-pruners `opus/`; katgpt-core `mcts`, `caddtree_budget` |
| Acceptance/confidence calibration | Bebop `α ≈ a−bH` marginal fit (R243); STS chain-rule calibration (R316 distilled item 2 — **not yet shipped**) | katgpt-core `bebop` |
| OOD-proportional compute | `RiskControlledExit` signal-ensembling (Plan 575 — picks the most efficient feasible stopping signal); density-aware scheduling (R350 — **density axis, not OOD axis**) | katgpt-core Plan 575 |
| Learned energy head | (none — training) → riir-train Plan 388; GenRM generative verifier (riir-train Plan 342) is the generative sibling | riir-train |
| Convex-landscape mode-merging failure | (no shipped detector; no live consumer — see row J) | — |

### 2.2 Path 0 inventory — coordinator merge of the adversarial panel

Both advocates ran (No-GD + Model-based, same parallel batch as the §4 searches). Per-row disposition — every row ends in (a) filed plan, (b) issue, or (c) audited discard with mechanism-level reason:

| # | Component | Track | Disposition |
|---|---|---|---|
| A | Trained energy head `E_θ(x,ŷ)` | Training | **(a) → riir-train Plan 388.** R346 §2.3 audit carries: Path 1 fails (no state to correct — learned-from-scratch function), Path 2 fails (no base weight to perturb), Path 3 fails (Dirichlet/Hodge/spectral degenerate to smoothing). What requires GD: fitting a scalar compatibility function to outcome labels. |
| B | Optimization-training loop (HVP backprop) | Training | **(c) audited discard as separate work.** Plan 388 trains a *discriminative* head (logistic on accept/reject outcomes), NOT optimization-training — the 3.33×/step HVP path needs the full EBT architecture and a from-scratch model (parked, row P). Mechanism-level reason: our consumer (acceptance calibration) needs a scorer, not a generator. |
| C | Landscape regularization (replay buffer, Langevin, randomized α/steps) | Training | **(a) → Plan 388 recipe section, SELECTIVE.** S1→S2 curriculum + per-element randomized α + randomized 2–3 steps port to any iterative training. Replay/Langevin are optimization-training-specific — porting them to LoRA/quest-grammar training without an energy objective is cargo cult (advocate's warning, adopted). |
| D | S1→S2 stability curriculum | Training | **(a) → Plan 388** as the general stability recipe for looped/iterative LoRA training (edge_lora topology, any unrolled loop). |
| E | Energy-scalar-as-uncertainty (candidate-conditional) | Modelless | **(c) audited discard as a new primitive.** Signal-diff: `contrastive_scope` consumes INPUT-side D-statistic (is this input in-distribution) — not candidate-conditional compatibility; the fusion margin consumes candidate scores but over a DISCRETE fixed goal set, not a refinement trajectory; Plan 575's `RiskControlledExit` consumes any monotone confidence trajectory and needs exactly such a signal. Coverage: the signal SHAPE ships for discrete selection; the continuous-refinement consumer does not exist in the stack. Recorded as a **candidate signal for Plan 575 consumers** — filing a primitive with no consumer violates the no-default-consumer rule. |
| F | Plateau/convergence early stopping | Modelless | **(c) audited discard — covered.** `gain_cost_halt` (R266, damped fixed-point halting), R344, R525, and riir-games deliberation_cadence's settled-early-commit all ship the law. EBT's energy-plateau is the same stop signal on a different scorer; no new mechanism. |
| G | OOD-linear thinking-gain budget law | Modelless | **(c) audited discard as a primitive; recorded as a calibration input.** Signal-diff: R350's scheduling consumes DENSITY (population count) — a physically distinct axis from distribution-shift magnitude; Plan 575's exit framework consumes confidence trajectories, not shift magnitude. The paper's linear law (compute budget should scale with OOD magnitude) is a **candidate signal for Plan 575's signal-ensembling** (its own doc: "the paper's signal-ensembling picks the most efficient feasible stopping signal"). No live consumer beyond that → recorded here, not filed. |
| H | BoN + min-energy selection | Modelless | **(c) audited discard — covered.** BoMSampler (Plan 281), `boltzmann_probabilities` (R317), MCTS leaf eval. EBT's delta (per-prediction BoN on every token) is a deployment policy, not a mechanism. |
| I | Adversarial low-score candidate hazard | Modelless | **(c) audited discard — partial coverage + validating.** `evidence_tripwire` (engram, Bench 832) detects rank-inversion poisoning; the healer's strict-keep + Issue-030 relevance gate guard created-wrong. EBT adds a measured **scaling law**: adversarial minima shrink with data scale — direct support for the healer store densification direction (8,235 trajectories, 88.4% one-rule). Recorded; no new mechanism. |
| J | Multi-modal convex-landscape failure predictor | Modelless | **(c) audited discard — no consumer.** No production loop in the stack performs CONTINUOUS candidate-space refinement against a modelless score where mode-merging would bite (deliberation/MCTS are discrete-select; healer fixpoint is not score-refinement). A bimodality-of-top-k-scores detector would be a primitive awaiting its consumer — the Issue 528 no-premature-abstraction precedent. Recorded here; re-open when a latent-refinement consumer materializes. |
| K | Verify ≫ generate asymmetry | Modelless | **(c) audited discard — architecture principle already embodied** (ConstraintPruner, verify paths, strict-keep gates). EBT's contribution is the generalization argument (verifiers transfer OOD better), which matches our own measured OOD results. |
| P | From-scratch EBT pretraining (the +35% claim) | Training | **(c) audited discard — parked with reopen trigger** (advocate #5, adopted): testing a pretraining-scaling law needs ≥3 compute points × seeds (~200–400 4090-h at ~150M) with no waiting consumer. Reopen only if Plan 388's heads show a measured serving-side thinking gain. |

### 2.3 Novelty gate (per track)

- **Modelless track:** Q1 prior art — R346 + R317 + R266 + Plan 575 cover every signal shape (audited above); Q2 no new capability class; Q3 no selling point beyond shipped ones; Q4 n/a. **0/4 — not Super-GOAT. Pass.**
- **Training track:** engineering GOAT, not Super-GOAT — the class is published prior art (SpecDec++ arXiv:2405.19715; on-the-fly adaptation ACL 2025). The delta is ours: the measured Bench-696 calibration defect on our serving path, calibration against recorded accept/reject outcomes, the conformal floor (Report-the-Floor), and the EBT discriminative framing. **No novelty claim made or needed.**

### 2.4 Closest-cousin fusion (what paper × A × B produces)

- **EBT × R316 (DSpark) × Bench 696 traces** = calibrated acceptance-energy head replacing the overconfident Markov confidence head, wired through the confidence-scheduled decode path → **Plan 388 Phase 1** (the fusion that files).
- **EBT recipe × riir-train 342 (GenRM)** = the discriminative sibling of the generative L4 verifier — same seam, logistic-on-outcomes instead of generate-and-grade; cheaper inference (one scalar vs a generation). → Plan 388 Phase 3 cross-ref.
- **EBT uncertainty law × Plan 575 (RiskControlledExit)** = OOD-magnitude as a candidate stopping signal in the calibrated-exit ensembling. Recorded (row G).

---

## 3. Verdict

### Tier

- **Training track: Gain** → `riir-train/.plans/388_ebt_energy_head_dspark_calibration.md` (filed this session). Anchor: the measured DSpark overconfidence defect; <10 GPU-h Phase 1; conformal-floor gate; parked from-scratch pretraining with explicit reopen trigger.
- **Modelless track: Pass** — the modelless half is covered by R346's audit + R317 + R266 + Plan 575 + BoM; the two genuinely-unshipped deltas (OOD-budget law, mode-merging predictor) are audited-discarded for want of a consumer, recorded in §2.2 rows G/J.

### MOAT gate

- `katgpt-rs`: no new public primitive (modelless half ships). No plan.
- `riir-train`: active moat — Plan 388 filed (trained calibration artifact + recipe).
- `riir-clippy`: the fixer-verifier seam is owned by Plan 342 (GenRM); Plan 388 Phase 3 cross-refs it as the discriminative alternative. No separate issue.
- `riir-ai`: nothing to file — the game-runtime energy vocabulary (HLA/attractor/deliberation) already embodies the verify-refine-stop loop, and EBT validates it (per-token adaptive compute = per-NPC deliberation budgets; OOD-linear gains = fog-of-war deliberation spending).

### PASS-Redirects satisfied

R346 §5 updated with a pointer to this note (the predecessor note gains the successor's delta record). R316/R243/Plan 575 are cited here as the consumer chain for Plan 388.

---

## 4. Cross-references

- **Predecessor:** `katgpt-rs/.research/346_IRED_Energy_Diffusion_Reasoning.md` — method proven at small scale; R346 §2.3's paths-1–3 audit carries to this paper unchanged; this note adds scaling evidence, the recipe ablations, S1/S2, and the uncertainty/OOD laws.
- **Consumer chain for the training track:** `riir-train/.plans/388_ebt_energy_head_dspark_calibration.md`; `katgpt-rs/.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md` (scheduler consumes the head); riir-ai Bench 696 (the defect); `riir-train/.plans/342_genrm_l4_verifier_gate.md` (generative sibling); SpecDec++ arXiv:2405.19715 (published prior art for the class).
- **Calibrated-exit substrate:** katgpt-rs Plan 575 / `riir-ai/.research/339_per_npc_risk_controlled_think_budget_guide.md` — rows E/G land there as candidate signals if a consumer asks.
- **Halting family:** R266 (FPRM), R344, R525 — row F coverage.
- **Energy vocabulary precedents:** R317, R100, R466, `dirichlet.rs`.

---

## TL;DR (repeat for grep)

EBT = IRED scaled to transformers + optimization-training backprop + landscape-regularization recipe + S1/S2 curriculum; first architecture to out-scale Transformer++ (+35%); thinking gains +29% vs baseline and grow linearly with OOD; energy scalar = learned epistemic uncertainty. Modelless half ships (plateau-halting, BoN-min, calibrated exit); trained half → riir-train Plan 388 (DSpark confidence recalibration via discriminative energy head, SpecDec++ prior-art class, conformal-floor gated). Verdict: Gain (training) / Pass (modelless).
