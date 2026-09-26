# Research 592: Loop Growth, Boundary Operators, and Scaling Exponents

> **Status:** RECORD — GAIN verdict, two tracks filed (training recipe plan + modelless instruments)
> **Source:** [arXiv:2609.19107 "How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents"](https://arxiv.org/abs/2609.19107) — Chen, Vegesna, Dahal, Wilson (NYU + Q Labs), submitted 2026-09-16, v2 2026-09-17. Code: github.com/qlabs-eng/scaling-exponents
> **Date:** 2026-09-26
> **Related Research:** 073 (LT2 looped), 097 (training-free loop), 414 (loop stability / readout blind spot), 035 (attractor models), 048 (HRM-Text), 238 (Muon LoRA — Muon caveat recipient)
> **Related Plans:** 108 (LT2 default-on), 136 (tf_loop), 428 (loop_stability_fix), 304 (gain_cost_halt), riir-train 364 (GRT uniform depth sampling), riir-train 373 (sotaku trained artifact), riir-train 324 (Ouro looped pretraining on a non-loop-trained base — TERMINAL FAIL, the nearest training-side cousin), **riir-train 421 (this paper's training recipe — filed with this note; renumbered from 420 after a live same-worktree dual allocation)**
> **Classification:** Public

---

## TL;DR

Architectural interventions at pretraining — loop-count **growth** during training (K=2→4 at a tuned transition) and a **boundary operator** `BO(h,e) = Norm(h) + α·e` applied between core passes AND before the coda — modify the **scaling exponent** γ of compute-optimal loss, not just the constant: compute multipliers vs vanilla that GROW with scale (BO 1.12×→1.25×, tied loop-growth 1.36×, untied growth 1.30×→1.55× at 10¹⁸→10²⁰ FLOPs). In the **data-constrained multi-epoch regime** (the paper's small-end experiment, and our regime), the optimal loop count grows with compute and loop-count scaling beats parameter scaling by **2.2×** even against per-size tuned weight decay, because overfitting tracks **stored** parameters, not executed depth.

**Distilled for the stack:**
- **(c) training:** the loop-growth recipe (growth schedule + multi-epoch loop scaling + WD-flat-in-K + re-tune discipline) → `riir-train/.plans/421_loop_growth_recipe.md`. One 4090-day ≈ 5–6×10¹⁸ FLOPs lands ON the paper's lowest ladder rung; the multi-epoch result was measured in exactly the regime of the stack's looped-training lane (small models, limited corpus, forced multi-epoch — the Plan-364/373 trained-artifact lane; the from-scratch C13/C14 track is RETIRED per riir-train `bb90e9eb`/`e0019165`, so the plan re-bases there).
- **(a) modelless:** the **KL effective depth** instrument (logit lens) + loop-flatness spectrum + write-fraction probe → `katgpt-rs/.issues/898` — measured exit-threshold calibration for the looped runtime's hand-tuned `loop_min`/`loop_max`/halt knobs. The BO operator at inference is hazard-classed with Issue 568 (trained-model phenomenon) until a BO-trained checkpoint exists.
- **(b) fusion:** belief re-anchoring (renormalize + re-inject observation embedding between latent-iteration passes) — fusion idea, novelty TBD (§4).

---

## 1. Paper Core Findings

1. **Exponents, not just constants.** Conventional wisdom (Bansal 2022, Hestness 2017) held architecture moves only the scaling constant. Fitting per-architecture compute-optimal recipes (four stages: base HP at d8, tokens-per-stored-parameter, growth timing, GLR scaling rule) and training ladders to 10²⁰ FLOPs: fitted exponents γ = 0.111 (Vanilla) → 0.114 (boundary op alone) → 0.116 (tied growth) → 0.117 (untied growth); regression SE < 10⁻³. Compute multipliers over vanilla RISE with scale — the signature that separates exponent from constant effects.
2. **Boundary operator.** `BO(h,e) = Norm(h) + α·e` where e = prelude output (the embedded input). Normalizing lets every pass write at full relative weight (curse-of-depth fix); re-injecting e keeps every pass conditioned on the input. **The before-coda application matters**: removing it costs ~2×10⁻³ loss (Table 4: Loop-2 3.2704 vs no-coda-inj 3.2912); operator ablations show each component contributes (Fig 15a: full BO 1.34× vs norm-only/inj-only 1.17–1.19×).
3. **Model growth.** Start at K=2 core passes, grow to K=4 partway (target 4 best at every budget; ρ — post-growth token fraction — ≈0.17 tied / 0.27–0.32 untied, broad minima; fixed ρ=0.30 costs <0.002 vs fitted). Growth prefers more tokens per stored parameter (5 vanilla → 6 BO → 7–8 growth): train shallow longer, add depth late. Mechanism: a fixed-depth model pays for depth it does not yet need; the wasted fraction grows with depth, and compute-optimal models get deeper with compute.
4. **Untying improves only the constant** (~1.06–1.16×, flat in scale). Tied loop-growth keeps the exponent gain at vanilla parameter count — weight sharing costs a fixed compute factor at every scale, never a growing one.
5. **Multi-epoch data-constrained (100M tokens × 10 epochs):** optimal loop count grows with compute (1.4→6.7 across budgets); loop-count scaling beats parameter scaling **2.2×** even against per-size tuned weight decay; looping regularizes (overfitting tracks stored params, not executed depth — optimal WD rises with depth but is nearly flat in K); gains largest when weight decay is weak. Untied looping (same depth, more params) does NOT beat tuned Operator-1 — weight sharing, not depth, is what helps under repetition.
6. **KL effective depth** (logit lens: first block after the KL peak whose decode is within 2 nats of final output): grows with compute for all families; BO +24% (20→24 layers), growth +50% (24→36) at 10²⁰ FLOPs. Loop-count scaling raises it faster than depth scaling at K=1.
7. **Random recurrence** (K~U{2..6} after growth, eval at 4): slightly worse than fixed growth on fresh data; flattens test-time pass sensitivity (+0.0003..+0.0028 at k=8 vs +0.008..+0.042 fixed-trained).
8. **Hyperparameter scaling is load-bearing:** transferring vanilla's tuned recipe to Operator-1 costs 7.8×10⁻³ loss AND **erases the exponent improvement**. GLR ∝ N^(−0.6..−0.8) per architecture.
9. **Muon × architecture:** Muon's compute-efficiency gain SHRINKS with scale under coupled width+depth scaling (constant gain under width-only) — optimizer-architecture interaction; relevant caveat to the stack's Muon LoRA lane (Research 238).
10. **Extrapolation validated:** a 7.4B Untied-Grow run at 8× the largest fitted compute lands on the predicted loss curve; matches GPT-3 13B on CORE at ~20× less compute (indicative, not controlled — different data + eval pipeline).

---

## 2. Path 0 inventory (three-track decomposition, advocates merged)

| # | Paper component | Track | Stack coverage (signal-diff checked) | Disposition |
|---|---|---|---|---|
| 1 | γ-exponent modification, compute multipliers 1.12→1.55× | c | none — no pretraining ladder; one 4090-day ≈ 5–6×10¹⁸ FLOPs, two compute points cannot resolve γ deltas of 0.003–0.006 fitted over two decades | Plan 421 measures **constant-floor only** + directional KL-depth mechanism witness; exponent claims recorded scale-mismatched |
| 2 | Boundary operator incl. before-coda placement | a+c | Plan 428 ships inter-loop RMSNorm = the Norm(h) half, WITHOUT α·e re-injection, NOT before the readout; Issue 568 shows injection phenomena are trained-model phenomena | (a) inference arm **contingent on a BO-trained checkpoint** (Plan 421 output) with Issue-568 refutation expectation; (c) in the training architecture (Plan 421 P4) |
| 3 | Growth schedule K=2→4, ρ≈0.17–0.32, target 4 | c | GRT / riir-train Plan 364 = uniform sampling r~U{1..R} from scratch — a different schedule the paper itself MEASURES (slightly worse fresh-data than growth) | Plan 421 P2 (growth) + P3 (growth-then-sample as the Plan-364 synthesis) |
| 4 | Multi-epoch loop-count scaling, WD flat in K | c | none — the looped-training lane's runs are multi-epoch data-constrained by construction | **Plan 421 P1 — the regime-matched headline** |
| 5 | Untied weights | c | LoopMode::WeightShared default-on; untying = K× stored params | **Declined (R6)**: constant-only 1.06–1.16× for K× memory on 24 GB; revisit only if serving ever demands K-distinct weights |
| 6 | KL effective depth diagnostic | a | SHIPPED 2026-09-26 as opt-in `kl_depth_probe` ([Bench 899](../.benchmarks/899_kl_effective_depth_goat.md): G1 PASS on micro fixtures, no default change); before that: not shipped; prior art established (HRM-Text 2605.20613 runs it; §4 search confirmed derivative) — our claim is the **exit-threshold calibration instrument**, not the diagnostic | katgpt-rs `.issues/898` |
| 7 | Test-time pass-sensitivity spectrum | a+c | ELT loop_min/loop_max exist (hand-tuned); spectrum unmeasured; GRT owns the training half | `.issues/898` (spectrum) + Plan 421 P3 (flattening recipe) |
| 8 | Write-fraction spectrum (curse-of-depth probe) → ε halt calibration | a | gain_cost_halt ships absolute step-size + angular signals; ε is hand-set | `.issues/898` sibling measurement |
| 9 | GLR/TPP/per-arch tuning laws | c | no N-scaling ladder; the tuning-discipline half transfers | Plan 421 gate discipline (R9): **inherited-HP GOAT gates are void** |
| 10 | Muon × depth interaction | c | Muon LoRA lane (Research 238) | recorded caveat; A/B deprioritized (Plan 421 P5, discard reason below) |
| 11 | BO latent analog: belief re-anchoring | b | decay_confidence / goal_salience substrate dense; Issue-568 hazard applies to untrained kernels | fusion idea, novelty TBD — riir-ai issue deferred (worktree contested at filing time, §4) |

**Auditable discard/scoping reasons (§3.5):**
- *R3 exponent claim at our scale* → scoped to constant-floor: the fitted γ deltas (0.003–0.006) require a compute ladder we cannot run; two compute points measure a chord, not an exponent. The paper's own mechanism witnesses (KL effective depth) remain measurable and gate the direction.
- *R7 Muon A/B* → deprioritized (sequencing, not merit): swapping the optimizer while the architecture schedule is unsettled confounds both; the paper's direction (gain largest at small scale) predicts our scale is the favorable end, making the A/B low-information until the loop-growth gate lands. Re-arm: after P1–P3.
- *No-GD item 8 (BO at inference) as an independent primitive* → demoted to a contingent follow-up arm: the Issue-568 PoC measured NO TRANSFER for input injection on a substrate that could not do the task, and the generalized lesson — injection phenomena are trained-model phenomena — applies to any checkpoint not trained with BO. Applying it blind to frozen checkpoints is a refutation-expectation experiment, not a primitive.
- *No-GD item 6 (K=4 as config default)* → scoped to "testable prior" inside the instrument issue: the K*=4 constant is the paper's elbow on THEIR ladders; our constant, if different, is recorded in the issue, never promoted without our own sweep.

---

## 3. Distillation per track

### 3.1 Track (c) — training recipe → `riir-train/.plans/421_loop_growth_recipe.md`

The regime-match card: one 4090-day ≈ 5–6×10¹⁸ FLOPs (165 TFLOPS bf16 peak × ~40% MFU) sits ON the paper's lowest ladder rung, and the multi-epoch data-constrained experiment (100M × 10 epochs, small model, overfitting bound by stored params) is the paper's small-end result — measured where the stack's looped-training lane trains (Plan 364/373: GRT uniform-depth sampling, sotaku trained looped artifact). The from-scratch C13/C14 SFT track is RETIRED (riir-train `bb90e9eb` "from-scratch + Ouro tracks dead", `e0019165` "C13/C14 retired") — the plan re-bases on the live looped-training lane and a fresh dense-arch probe model, NOT on C13 (a stale premise two sessions repeated from the daily-check skill). Recipe items with GPU-hours (staged, owner-gated — riir-train is lowest repo priority and the 4090 window is owner-scheduled):

- **P1 (headline, ~3 GPU-days):** loop-count scaling under multi-epoch teacher-data constraints. Grid K ∈ {1,4} × epochs ∈ {E,2E} + one param-grown control; WD pinned at the K=1-tuned value (R5 is the control condition AND the free transfer rule). GOAT: K-scaled run at 2× compute ≤ param-grown control (the 2.2× ratio, measured not assumed) + the lane's own quality metric not regressed. **Plan 324's terminal FAIL is accounted in the gate: 324 grew looping ON a non-loop-trained base (the Issue-568 hazard class) and collapsed; P1 trains the loop in from step 0 — in-distribution from the first step — which is both the paper's setting and the signal-diff that explains 324's failure mode.**
- **P2 (~1.5–2 days):** growth schedule K=2→4 at fixed ρ=0.30 (the broad minimum). GOAT: growth ≤ fixed-K=4 from scratch at matched FLOPs; |fixed-ρ − argmin| ≤ 0.002.
- **P3 (~1.5 days):** growth-then-random-recurrence (the Plan-364 synthesis): K=2→4 growth, then tail sampling U{2..6}. GOAT: pass-sensitivity spread ≤ ½ of fixed-growth's, fresh-data penalty ≤ 0.005, early-exit quality ≥ the Plan-364 artifact. This gate is what makes the serving-side `loop_count` knob trustworthy across its range.
- **P4 (~2.5–3 days):** boundary operator in the training architecture, **constant-floor claim only** (≥5–10% FLOP reduction at matched loss), with the R9 discipline: mandatory LR/WD re-grid — an inherited-recipe gate is void (the paper: transferred recipe costs 7.8×10⁻³ and erases the gain).
- **P5 (deprioritized):** Muon A/B after the loop-growth gate.

**Paths 1–3 exhausted before filing (modelless unblock):** (1) freeze/thaw — N/A, mechanisms are architectural/schedule, not bias corrections; (2) raw/lora hot-swap — a deterministically constructed adapter cannot create loop-competence (looping requires weight-tied trained cores); (3) latent-space correction — BO is latent-expressible but out-of-distribution without BO-trained weights (Issue 568; Plan 324's terminal collapse on a non-loop-trained base is the in-tree confirmation). The modelless yield is real but lives in the instruments (§3.2) — the dual-track contribution is recorded.

### 3.2 Track (a) — modelless instruments → `katgpt-rs/.issues/898`

Four offline measurements over checkpoints the stack already serves, feeding knobs that are hand-tuned today:
1. **KL effective depth** — per-checkpoint scalar; the natural exit point for ELT.
2. **Loop-flatness score** — spread of L(k) over k ∈ {2..8}; checkpoint-selection criterion for loop-count-elastic serving.
3. **Write-fraction spectrum** — ‖Δh_k‖/‖h_k‖ decay across loops; principled ε for gain_cost_halt.
4. **ELT depth-execute distribution** — histogram over an any-time-exit serving pass; the runtime mirror of ρ.

Output: **depth-calibrated loop exits** — fit thresholds on fixture half A, predict optimal exits on half B within ±1 loop, holdout-validated, loss parity vs hand-tuned at matched compute, kill-switch bit-identity. GOAT gate = the holdout prediction; a gate without that arm is unvalidated extrapolation (the paper's own 8×-extrapolation protocol, adopted as a gate arm).

### 3.3 Track (b) — fusion idea (novelty TBD)

**Belief re-anchoring:** between belief-evolution / MCTS-iteration / consolidation passes, renormalize the belief state and re-inject the observation embedding, sigmoid-gated (never softmax) — BO's shape transported to latent iteration loops. Think-brain only, never crosses the raw sync boundary. **Cautions:** Issue 568 measured NO TRANSFER for input injection on an untrained kernel (the regime split is a trained-model phenomenon); the riir-ai cognition surface is dense (goal_salience, decay_confidence) — substrate-first audit required before any plan. riir-ai issue deferred at filing time: that worktree was mid-operation (reflog permission denied, 6 dirty files, HEAD behind origin — a concurrent session). File `.issues/1009` (or next) in riir-ai when quiescent.

---

## 4. Cousins and why they do not kill

1. **GRT / riir-train Plan 364** (uniform depth sampling r~U{1..R}): different schedule the paper itself measures — random recurrence is slightly WORSE than fixed growth on fresh data, better only at test-time pass tolerance. Plan 364's objective is emergent early exit; the paper's is the growth transition (ρ) + growth target + the multi-epoch capacity axis. Extension, not duplicate: P3 fuses them (growth-then-sample), and the paper prices the tradeoff Plan 364 left open.
2. **Plan 428 / Research 414** (inter-loop RMSNorm, inference-time): ships the Norm(h) half only — no α·e re-injection, no readout-boundary application, no training recipe, no scaling claim. The paper's Table 4 (coda injection worth ~2×10⁻³) and Fig 15a (each operator component contributes) pin exactly what our stack has not tested. Signal-diff: scale-removal-between-iterations (ours) vs scale-removal + input-conditioning + readout boundary (paper).
3. **Issue 568** (input injection NO TRANSFER): hazards the modelless BO-at-inference extraction — disposition is contingency (§2 row 2), not abandonment: the PoC source stays as a permanent regression check and re-arms the moment a BO-trained checkpoint exists.
4. **HRM-Text / Research 48**: the KL-depth diagnostic is established prior art (confirmed by §4 search); our novelty is the **calibration instrument** — measured thresholds replacing hand-tuned ELT/halt defaults — which no existing note or plan owns.
5. **riir-train Plan 324** (Ouro looped pretraining on the C13 base — TERMINAL gate-collapse FAIL): the nearest training-side cousin, and the one whose failure mode most informs this verdict. 324 applied looped pretraining ON TOP of a non-loop-trained base — the Issue-568 hazard class — and its G-OURO gates collapsed. This paper's setting differs at the root: looping is trained in from step 0 (weight-tied cores, boundary operator in-distribution from the first batch), and the multi-epoch regime result is a fresh-data-repetition axis 324 never tested. Plan 421 inherits 324's lessons as preconditions (dense/uniform arch — `tf_loop` is documented incompatible with Kimi-K3's hybrid layers; stability knobs armed from step 0; the exit gate G-OURO-1a collapse detector), and P1's gate accounts for the FAIL rather than re-running it blind.

---

## 5. Verdict

**Overall: GAIN** (GOAT-tier components; no new capability class).

- **(c) training: GAIN → `riir-train/.plans/421_loop_growth_recipe.md`** — a staged, gate-per-phase recipe plan; Path 0.5 (applicable training paper gets a plan, not a redirect). Owner-gated scheduling (riir-train lowest priority; 4090 window owner-scheduled).
- **(a) modelless: GAIN → `katgpt-rs/.issues/898`** — the instrument set + depth-calibrated exits; GOAT = holdout exit prediction.
- **(b) fusion: novelty TBD** — recorded in §3.3; riir-ai issue to follow.

Not Super-GOAT: the exponent claim is the paper's, unverifiable at our scale; our extractions improve existing primitives (loop runtime config, training recipes) rather than create a capability class. No "candidate" hedge — the two GAIN tracks are filed this session.

**MOAT gate:** katgpt-rs — in scope (the looped-transformer stack is a katgpt-rs runtime primitive family: Plans 108/136/428, Research 073/097/414 all live here; instruments serve its default-on looped runtime). riir-train — in scope (active moat: training-method implementations + configs; Path 0.5 satisfied with recipe + GPU-hours + GOAT-vs-modelless gate).

**Gate-design law adopted (has enforcement teeth):** every architecture-mode GOAT must (a) retune config for the mode, (b) carry the transferred-recipe negative control (expected to underperform — proving the tuned arm is not luck), (c) report the compute-multiplier *direction* at matched loss, not just point gains. Adopted into Plan 421's gate discipline and Issue 898's calibration gate. A mode GOAT missing the negative control fails review.

## Numbering note (collision record)

This note was first filed as `.research/591` and the training plan as `riir-train/.plans/420` — a same-worktree dual allocation with a concurrent session (their `591_Trajectory_Aligned_Curiosity.md` + `420_self_play_zero_data_pretraining.md`, landed 11:33–11:35 same morning; the dual-allocation gate cannot see same-worktree races, only upstream divergence). Resolved by citation-weight asymmetry: the sibling's 591 is the hub of a four-file citation constellation (`riir-train/.research/458`, their plan, `riir-ai/.research/389`, `katgpt-rs/.plans/610`) while ours was cited only from this session's own files — **we moved**: 591→592, 420→421. Both highwaters re-bumped. Recorded here because the collision is invisible to `scripts/dual_allocation_gate.py` by construction and `scripts/citation_weight.py`'s strict-margin rule would have declined to arbitrate a tie.

## Summary

**(1) Original task:** Run the research workflow on arXiv:2609.19107 — distill, classify, verdict, file.

**(2) Accomplished:** Full workflow executed: internal-first grep (no prior distillation), five-step pre-flight, three-track adversarial panel (No-GD + Model-based advocates + §4 prior-art web search), Path 0 inventory with auditable scoping reasons, signal-diffs against all five cousins (incl. riir-train Plan 324's terminal FAIL, added at verdict round 2), dual-allocation gate + a LIVE same-worktree collision resolved by renumbering (591→592, 420→421), files written: `katgpt-rs/.research/592_Loop_Growth_Scaling_Exponents.md` (this note), `riir-train/.plans/421_loop_growth_recipe.md` (staged training recipe, 12–18 GPU-days, owner-gated), `katgpt-rs/.issues/898_kl_effective_depth_loop_exit_calibration.md` (instruments + contingent BO arm), back-reference added to `.research/414`, highwaters bumped (592 / 898 / 421).

**(3) What remains:** riir-ai fusion issue (belief re-anchoring) deferred — worktree contested at filing time; file when quiescent. Plan 421 execution is owner-gated (riir-train lowest priority; 4090 window owner-scheduled). Issue 898 implementation is a normal katgpt-rs work item (unassigned). Exponent-tier claims recorded scale-mismatched by design — revisit only if a pretraining ladder ever exists in the stack.

**(4) Active plan state:** `riir-train/.plans/421_loop_growth_recipe.md` — filed, phases unchecked (P0 preconditions → P5 Muon, owner-gated). `katgpt-rs/.issues/898` — open. This note RECORD. §5 verdict ping-pong completed before commit (see git log for the verdict record).
