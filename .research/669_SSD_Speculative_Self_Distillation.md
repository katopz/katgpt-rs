# Research 669: SSD — Speculative Self-Distillation (Mixed-Policy Rollout Steering)

> **Paper:** [Speculative Self-Distillation enables Efficient Knowledge Internalization](https://arxiv.org/abs/2611.12016) — Talaei, Bhatia, Soltani Moakher, Hübotter, Saberi, Mirhoseini (Stanford/ETH), NeurIPS 2026, submitted 2026-11-09
> **Code:** https://github.com/ShayanTalaei/SpeculativeSelfDistillation — Apache-2.0, pinned `@ cccf657b6e67fb6f9a7f5353bf8062cc2bd16892` (clone verified before verdict; `.raw/ssd` deleted at close)
> **Date:** 2026-11-13
> **Verdict:** ⭐ GOAT (track c — training) → riir-train Plan 496 · fusion ideas (track a — modelless) → katgpt-rs Issue 1210 · track b discard
> **Not Super-GOAT:** Q1 fails on class-level published prior art (SKD ICLR 2025 / AdaSwitch 2510.07842 — per-token switching is established for two-model distillation); Q3 partial (enabling, not a product selling point)
> **Relates To:** Research 038 (SDAR), 036 (ROPD), 217 (TRD/TRDraft), Plans 120 (VPD) / 121 (RMSD) / 169 (GDSD), Research 316 (DSpark confidence-scheduled spec decode), riir-refine `seat_rerank` escalation seam

## TL;DR

Self-distillation internalizes privileged context into weights (teacher = same model + context in prompt; student = unconditioned). SSD makes the **rollout policy** per-token mixed: the student generates by default, and the teacher takes over exactly at **inflection positions** — where teacher/student next-token divergence (JSD or KL) exceeds a threshold τ. The dense forward-KL distillation loss still applies at EVERY position; the switch changes only who emits the token ("selective trajectory correction, not selective supervision"). One dial τ interpolates off-policy (τ→0) ↔ on-policy (τ→∞); JSD is bounded by ln 2 so any larger τ is exactly on-policy. Results: on-policy-level accuracy at 38–69% fewer supervised tokens (Qwen3-4B/2.5-7B/3-14B), 69% less training time with a paired dual-GPU engine (92% of student-only throughput), least forgetting (avg general-benchmark Δ −1.38 vs −4.47 on-policy / −6.6 off-policy), and teacher contribution auto-decays 43%→1–5% with no schedule. The stack verdict: the training recipe is GOAT for the draft-model/planner distill lanes (Plan 496); the decision/measurement/transport math decomposes into modelless extractions (Issue 1210); the decay mechanism itself is honestly GD-bound.

## Pinned claim (§1.5 precondition)

"Per-token divergence-threshold **rollout-policy switching** for **privileged-context self-distillation**, consuming the teacher-vs-student next-token divergence (JSD/KL) as an inflection signal, distinguished from SKD/AdaSwitch/SWITCH (per-token switching for ordinary two-model distillation) and GATES (selective privileged-context distillation gating *questions/trajectories*, not the rollout policy) by applying the switch to the self-distillation setting where divergence has the privileged-context reading 'where the context changes the prediction' — plus the empirical laws (signal starvation along on-policy rollouts, auto-decay, the accuracy-efficiency-forgetting Pareto frontier)."

## §4 Prior art (run before the verdict — what the paper itself concedes + what the searches added)

The paper's own Related Work names the class:
- **SKD** (arXiv:2410.11325, Xu et al., ICLR 2025, ~86 citations) — "Speculative Knowledge Distillation": interleaved sampling between teacher and student, token-level switching for two-model distillation.
- **AdaSwitch** (arXiv:2510.07842, Oct 2025) — student explores within its capability, teacher guidance integrated "only when divergence" (sliding-window KL threshold) — two-model distillation.
- **SWITCH** (Koo et al., 2025) — per-position divergence criterion — two-model distillation.
- **GATES** (Stein et al., 2026) — closest in SETTING (privileged-context self-distillation made selective) but gates which questions and tutor trajectories are distilled via agreement among context-conditioned tutor rollouts — NOT which policy generates the rollout.
- Sequence-level mixing lineage: GKD (Agarwal et al. 2024), DAgger (Ross 2011), scheduled sampling (Bengio 2015), imitation KD (Lin 2020).
- OPSD landscape is dense in 2026: "Adaptive Teacher Exposure for Self-Distillation" (arXiv:2605.11458), "What Should a Self-Teacher See?" (arXiv:2609.25623), "Position-Weighted OPSD" (arXiv:2609.20612), "A Token-Level Analysis of OPSD" (OpenReview), SDFT (Shenfeld et al., ICML 2026 — the paper's on-policy baseline and codebase parent).

**Novelty ruling:** the *mechanism class* (per-token divergence-gated rollout switching) is published prior art; SSD's contribution is the privileged-context self-distillation application + the supervision-signal diagnosis + the empirical laws. Internal searches found no identical prior doing per-token policy switching in the privileged-context self-distillation setting — the claim survives *qualified*.

## Method distillation (pinned in code)

- `src/ssd/strategies.py` — the routing family over raw logits: `PureStudent` (on-policy) / `PureTeacher` (off-policy) / `KLSwitch` (forward/reverse, per-direction semantics documented) / `JSDSwitch` (symmetric, range [0, log 2] nats — verified in docstring + `config.py` pins default τ=0.5 inside it) / `EntropySwitch` / `TopKIntersection` / `LogitFusion`; every decision returns a `TokenDecision` provenance row (source + divergence + reason).
- `src/ssd/loss.py` — dense forward-KL at every position + provenance masks (`all` default / `teacher_only` / `student_only` / directional as ablations).
- `src/ssd/teacher_cache.py` — **the modelless crown jewel (verified line-by-line)**: store teacher *raw* pre-temperature logits at union(topK(teacher), topK(student)) indices + the full-vocab `logsumexp` at the cache temperature; kept entries recover exact probabilities (`log q(v) = logits[v]/T − lse`); the dropped tail folds into one "everything else" bucket; KL between the two coarsened distributions is, by the log-sum inequality, a **LOWER BOUND** on the true full-vocab KL, tightening with K; `kept_mass` reports coverage. Union-of-both-top-Ks is load-bearing (direction-weighted divergences truncate exactly the terms the non-weighted side is made of). Numerics: `EMPTY_SLOT_LOGPROB = -1e9` (finite sentinel; `0·inf=NaN`), `_TAIL_EPS = 1e-12` (fp32 clamp), raw-logits-not-logprobs ("a log-prob captured at one temperature cannot be rescaled to another").
- `src/ssd/config.py` — `teacher_cache ∈ {off, verify, on}` (verify = run both + log reproduction error), `paired_transfer_logits=True` (fp32 logits) vs hidden-state transfer projected through a resident teacher-head copy; `paired_teacher_cache="hidden"` (GPU-resident teacher hidden states for the loss).
- `src/ssd/distil_trainer.py` — the one-rollout-per-optimizer-step invariant (`steps_per_generation == gradient_accumulation_steps, num_iterations == 1`) and its staleness warning: a generation buffer outliving an optimizer step describes a teacher one or more EMA updates behind what the loss would recompute.
- `paired_vllm/` — two-GPU lockstep engine: CUDA-IPC zero-copy weight refresh, teacher sends final post-norm hidden state (2,560 floats for Qwen3-4B) instead of full logits (151,936 → ~59×), routing/sampling/token exchange stay on-GPU, tokens read back every 8 steps; joint decode ≈ 92% of student-only throughput.

## Path 0 decomposition (both advocates merged; coordinator spot-checked the code facts)

| # | Paper component | Modelless-shippable? | Extraction / route |
|---|---|---|---|
| 1 | D_switch > τ routing family | YES — softmax + one reduction + one compare | Issue 1210-C1 (gate family + diagnostics); coverage signal-diff vs `seat_rerank`/ESC escalation done: those gate on embedding-distance/lane-quality at the DECISION level, not token-distribution divergence at GENERATION positions — uncovered where an actual generative student exists |
| 2 | Inflection provenance substrate | YES | Issue 1210-C2 (receipts + failure-prediction AUC gate) |
| 3 | Positional supervision-signal diagnostic | YES as instrument; 3.3×/17× numbers are paper-only, re-measure locally | Issue 1210-C3 |
| 4 | τ-interpolation law + JSD ≤ ln 2 bound; post-hoc τ-sweeps on logged divergences | YES — on a frozen two-lane capture the whole sweep is post-hoc (stationary divergence field) | Issue 1210-C4 (escalation-band sweep) |
| 5 | Hidden-state-vs-logits transfer (~59×) + resident-head projection | YES (frozen serving posture needs no refresh; head = BLAKE3-pinned artifact) | Issue 1210-C5 pointer; nearest stack surface = riir-infer plan-623 multi-host wire (staged/opt-in — not a plan today) |
| 6 | Sparse-KL **lower bound** (top-K union + full-vocab lse + kept_mass) | YES — closed-form inequality + capture format, zero GD | Issue 1210-C6 (the strongest extraction; O(K) bounded divergence with known error sign) |
| 7 | Union-of-top-K asymmetry law | YES | folds into C6 |
| 8 | Dense divergence field + provenance-directional asymmetry census | measurement halves YES; the GD update NO | C3 rider |
| 9 | Auto-decay (43%→1–5%) | instrument YES (provenance histogram); **mechanism GD-bound** (frozen student ⇒ stationary divergence field) | C2 rider + honest boundary |
| 10 | Internalization / 38–69% token saving / forgetting Δ | **NO — training outcomes** | → Plan 496 (track c) |
| 11 | Signal-starvation ordering law | YES as measurable ordering; cognition reading: conditioning influence washes out along a rollout — independent external support for stale-belief sigmoid-decay in the game lane | C3 note |
| 12 | Paired lockstep engine + zero-copy refresh | serving engineering; SKIP at draft scale (in-process loop) | Plan 496 infra table |
| 13 | Verify-cache posture + probability-weighted cache error | YES | C7 (validation doctrine) |
| 14 | Numerics micro-laws (finite sentinel, tail clamp, raw-logits capture) | YES | C6 riders (property tests) |

**Honest GD boundary:** the internalization effect itself, the decay mechanism, and the efficiency/forgetting gains require weight updates — claimed nowhere modellessly. Everything else is closed-form math over two distributions.

## Fusion map — closest stack cousins and the signal-diffs

- **SDAR (Research 038) / RMSD (Plan 121) — LOSS-side token gating.** Signal: advantage/reward gap (SDAR) or logprob+judge relevance (RMSD) → gates the LOSS weight. SSD: teacher-student distribution divergence → gates the ROLLOUT policy; loss stays dense everywhere. Different signal, different surface (loss vs trajectory). The stack's own recorded lesson: RMSD/GDSD were negative *modellessly in arena* with "infrastructure reusable for gradient-based path" — SSD is that gradient-based path, with the steering moved to the trajectory where RMSD moved it to the loss.
- **TRD/TRDraft (Research 217) — inference-time trajectory refinement** ("fix the trajectory, not the loss"). SSD is the training-time mirror: rescue the trajectory at inflection positions during rollout generation. Both compose with the loss-side lanes.
- **DSpark confidence-scheduled spec decode (Research 316) / LeviathanVerifier — inference acceptance.** Signal: confidence/acceptance probability AFTER drafting. SSD: divergence BEFORE generation at training time. The draft-model lane (Plan 496 T2) is where the two meet: training the draft exactly at its deployment rejection positions.
- **riir-refine `seat_rerank` / rethink ESC escalation — decision-level takeover.** Those compare lanes at the DECISION level (options echoed, pick-or-abstain; gate rho distance-only). SSD's gate is per-token inside a generative rollout. Uncovered where the student is actually generative; exactly analogous where it is not (τ band ↔ escalation rate band).
- **riir-infer `drift_guard` (Issue 1182) — serving divergence gate.** Signal: sampled rel-RMS of hidden activations vs exact twin. SSD/teacher-cache: output-distribution divergence over a top-K partition with a proven lower bound. Adjacent axes, not covered; C6 notes the drift_guard posture as the reference for any future wired consumer.

## Step-4 reframes

- **Game-context (riir-ai):** "teacher takes over where privileged information changes the action" is the two-brain bridge's existing shape (belief gates behavior via sigmoid within visible_radius; stale beliefs decay). SSD's delta is weight updates — the game lane gains a measurement instrument (conditioning-washout diagnostic, C3), not a new mechanism. Discard as a game filing; keep the note pointer.
- **Consumer-context (riir-refine rulewright):** the rulewright is modelless with no generative LLM in the default build (hosted_expert opt-in, abstain-first). No rollout to steer; the escalation-band sweep (C4) is the only transferable instrument and it is recorded in Issue 1210, not wired. No refine filing.

## Track verdicts

- **Track (a) modelless inference — fusion ideas → katgpt-rs Issue 1210.** Seven extraction candidates (C1–C7) each with a proof path (fixture + falsification arm + pre-registered gate); consumers are staged (multi-host wire, escalation seams, draft harness), so no GOAT claim is made and no primitive is promoted — ungoat discipline: runnable proof or `.issues/`, never a GOAT note for the record.
- **Track (b) self-adaptive runtime — DISCARD.** The decay mechanism needs mutated weights; the modelless residue (escalation-rate decay on corpus folding, a dynamics signal on genuinely mutated state) is recorded as C2's rider. One-line auditable reason: a frozen student's divergence field is stationary by construction, so the paper's central dynamic cannot ship without GD or corpus-mutation.
- **Track (c) model-based training — GOAT → riir-train Plan 496.** The incumbent capture-once posture is literally the paper's `PureTeacherStrategy` — its WORST arm on forgetting (−6.6 vs −1.38). The recipe (divergence-gated mixed-policy rollouts + dense FKL + EMA teacher + provenance) ports to the draft-model LoRA lane with a deployment-meaningful gate (acceptance rate), and the grammar lane is the cheapest exact-KL harness proof (closed vocab ⇒ the top-K lower bound is near-exact). GPU-hours: 1–3 (grammar harness) → 5–25 (draft cell, target-size dependent) on the 4090; fits one session each.

## References

- SSD paper + code (pin above); SDFT (Shenfeld et al., ICML 2026 — baseline + codebase parent)
- SKD arXiv:2410.11325 · AdaSwitch arXiv:2510.07842 · AdaSPEC arXiv:2510.19779 · GATES (Stein et al. 2026) · GKD (Agarwal et al. 2024)
- OPSD studies: arXiv:2605.11458 · arXiv:2609.25623 · arXiv:2609.20612
- Stack: Research 036/038/217/316 · Plans 120/121/169 · riir-train Plans 492/495 · riir-refine Plan 202 (seat_rerank) · riir-infer Issue 1182 (drift_guard), plan-623 multi-host
