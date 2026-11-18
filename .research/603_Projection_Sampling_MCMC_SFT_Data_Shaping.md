# Research 603: Projection Sampling — MCMC Data Shaping Makes SFT Rival RL

> **Source:** [Finetuning with Sampling: SFT Learns Better Than You Think](https://arxiv.org/abs/2610.02140) — Aayush Karan, Sitan Chen, Yilun Du (Harvard), arXiv:2610.02140, submitted 2026-10-01, NeurIPS 2026. [Blog](https://aakaran.github.io/finetuning_with_sampling/) · [Code](https://github.com/aakaran/finetuning-with-sampling) `@ 6d3e9f0bfaa98dcca534247dd35dc1b33dd8c428` (shallow clone, quotes verified at pin; clone deleted after distill)
> **Date:** 2026-10-03
> **Status:** Done — **Gain** verdict. Training track: GOAT → [riir-train Plan 438](../../riir-train/.plans/438_projection_sampling_sft_data_shaping.md). Modelless track: fusion ideas → `riir-reflex Issue 064` + audited discards below. No Super-GOAT.
> **Related Research:** 346 (IRED — energy-decrease acceptance, "Metropolis-without-the-reject-step" analogy; same author Yilun Du), 524 (GFlowNet — Metropolis/best-of-N as the modelless substitute, "generic MCMC" discard precedent), 558 (SLT/WBIC — MCMC-as-instrument precedent)
> **Related Plans:** riir-train 438 (projection-sampling SFT data shaping — filed this session), riir-train 426 T5 (the reflex corpus-synthesis lane — the closest structural cousin's design doc), riir-train 610 (the 4090 scoring queue issue)
> **Classification:** Public

---

## TL;DR

The paper reshapes **off-policy expert traces toward the base model's own distribution** before plain SFT, via a block-wise Metropolis loop: the same base model in chat mode with the expert solution in-context (the "self-teacher proposer") proposes suffix rewrites/extensions; a rewrite is **accepted iff the mean per-token log-likelihood of the full trace under the base model (completion mode, solution withheld) increases**. The accepted distribution approximates the **I-projection** `p_C ∝ p(x)·1(x∈C)` — the distribution closest to the base model consistent with the privileged information. SFT on projected traces rivals GRPO/OPSD/UFT on generalization + forgetting, and pass@k curves show **learning beyond distribution sharpening** (tasks go 0% → 67.2% pass rate). Framing: *sampling as a model-native operator that shapes data for learnability*.

**Distilled for the stack (modelless, inference-time):** the data-generation pipeline is 100% gradient-free as shipped — every moving part (privileged proposer, scorer, accept comparator, schedule) is inference-time compute plus one `<` comparison. The transferable primitive is the **universal data-curation operator**: *propose a constraint-preserving rewrite → accept iff the consumer's own density score improves → never inject content the proposer didn't produce*. The I-projection degrades gracefully under proxy densities: the KL *identity* is lost, the *ordering property* (minimal-deviation preference among constraint-satisfying candidates) is kept — and curation consumes ordering, not the distribution. The loss is measurable, not silent: the **minimal-deviation signature** (mean consumer-score drop of the accepted set vs the raw pool stays small) is a gateable fingerprint of doing I-projection rather than arbitrary filtering.

---

## 1. Paper Core Findings

### 1.1 The algorithm (verified from code at the pinned sha)

For each (question, expert solution) pair (`boost_sci.py` / `boost_math.py`, vLLM-backed):

1. **Block loop** (`block_num = 16`): each block first extends the trace by `jump_size = max_new_tokens/16` tokens sampled from the **self-teacher** (same base model, chat-templated with `PROBLEM + SOLUTION + current RESPONSE` in-context; temperature 0.6, repetition_penalty 1.05) — this extension is **accepted unconditionally** (privileged initialization; the irreducibility/liveness device).
2. **MCMC loop** (`mcmc_steps` per block): pick a random index `idx`; regenerate the suffix `gen[idx:]` from the teacher given `gen[:idx]` + solution; **accept iff `mean_loglik(prop) > mean_loglik(cur)`** under the base model in plain completion mode (question in-context, solution withheld). Early-stop on EOS.
3. The final accepted traces become SFT data (lr 5e-5, 2 epochs, batch 16, DFT codebase).

⚠ **The shipped acceptance is greedy ascent, not exact MH**: the teacher's token logprobs (the Hastings correction's numerator) are computed but the ratio is commented out (line 177; acceptance at line 203 is a bare `>` on mean-loglik). The paper's math story (MH targeting `p_C` with the constraint absorbed into the proposal) is the principled framing; the code ships greedy I-projection ascent. For data curation this is the *right* choice — dataset generation wants high-score artifacts, not unbiased draws; within-walk correctness needs only monotonicity (assertable), and diversity is supplied by the ensemble of independent walks at proposer temperature > 0. Exact `p_C` sampling (estimation, not curation) is a drop-in upgrade: sigmoid-tempered accept `σ(Δ/T)` with annealed T.

### 1.2 Results

- Chemistry (GPQA-style MCQ), competition math, medical QA on 7B-class models (Qwen2.5-Math-7B-Instruct, Olmo-3-7B-Instruct): SFT-with-sampling rivals GRPO/OPSD/UFT, often generalizing better and forgetting less.
- **Pass@k**: the projected-SFT curve stays above both base model and OPSD at large k — new capabilities, not sharpening. Individual tasks go 0% (pass@64 = 0) → 67.2% / 53.1%.
- Cost structure: `N_traces × 16 × (1 + mcmc_steps)` forwards, mostly short suffix regenerations; ≈35–40k decode-equivalent tokens/trace at L=1024. Sampling:SFT FLOPs ≈ 20–50×, but inference-only and batch-shaped.

### 1.3 Why it works (the mechanism we are buying)

Raw SFT on off-policy traces trains on-manifold-of-the-expert, off-manifold-of-the-learner → memorization + forgetting. The accept rule is a **distribution projection**: hill-climb each expert trace into the learner's high-density region while the teacher-initiated extension preserves task content; SFT then trains on-manifold. Quality is absorbed into the *proposal* (expert suffixes); the learner's likelihood is the only *acceptance* signal — no verifier, no reward model, no teacher model.

---

## 2. Path 0 decomposition (training-target math → component inventory)

| # | Component | Paper form | Stack coverage | Disposition |
|---|---|---|---|---|
| 1 | Privileged self-teacher proposer (conditioned generation) | same base model, chat mode, solution in-context | NO generative self-teacher ships (reflex synth's transplant is a deterministic proposer) | Plan 438: per-lane proposer impls (the base model IS resident in every training lane) |
| 2 | Learner-density scorer | mean per-token loglik, completion mode | **NO shipped code computes a learner-density gate** (skill_opt scores by benchmark; synth by teacher agreement; frontier_miner by oracle) | Plan 438 (per-lane scorers) + Issue 064 (reflex proxy-density version) |
| 3 | Accept rule (ascent over learner density, greedy) | `>` on mean-loglik | Skeleton ships (`skill_opt::ValidationGate` accept-iff-benchmark-delta>0) — **signal-diff: benchmark/task score, not density; skill doc, not corpus; no constraint-provenance audit** | Signal gap is real → Plan 438 + Issue 064 carry it |
| 4 | I-projection curation spec + minimal-deviation gate | `p_C ∝ p·1_C` | Not shipped | Plan 438 gates + Issue 064 |
| 5 | SFT consumption | DFT full-FT | Ships everywhere (grammar_training.rs, DualLeoTrainer, NLEH heads, RIDT distill) | Existing consumers; no new work |
| 6 | Healer fixture-mining density term | — | frontier_miner's acceptance invariant is oracle-ground-truth **by design** (unsolved-before/solved-after + exactly-once BLAKE3) | **Audited discard**: a density term would only rank among already-verified candidates; no documented retrieval-floor gap cited; injecting it fights the miner's own invariant |
| 7 | Memory-consolidation density-accept (neuron-db) | — | `free_energy_ledger` ΔF = evidence-cost − λ·ln n already ships an energy-style merge-accept | **Audited discard**: consolidation quality gap undocumented; ΔF rule covers the accept-rule shape; projection there is speculative without a consumer |
| 8 | pass@k + retention eval methodology | anti-sharpening discriminator | not standard in our lanes | Adopted into Plan 438 gates |

Every row lands in (a) Plan 438 / (b) Issue 064 / (c) audited discard with reason. No "candidate" residue.

---

## 3. Distillation

### 3.1 The universal curation operator (modelless)

```
propose(prefix, C) -> rewrite        // constraint-preserving by construction (provenance)
accept iff score_consumer(rewrite) > score_consumer(current)
never inject: acceptance can only reject, never add unproposed content
```

Instantiations across the stack's domains: code spans (compile/lint gate = the hard `1(x∈C)`; accept iff consumer diagnostic/rank score improves) · fix trajectories (template/fix-mode proposers; corpus-idiom density) · decision corpora (alternative action for the same state under the same information; calibrated confidence as score) · behavior traces (sim as proposer; frozen reward proxy as score) · memory/lesson consolidation (leave-one-out self-retrieval hit-rate as the density proxy).

### 3.2 I-projection under proxy densities

The hard-filter half is **exact and structural** (membership in C enforced by proposer provenance, not statistics). The density half degrades to **order-preservation** under any rank-consistent proxy — corpus likelihood (reflex Lz4FlexDrafter), retrieval kNN density, rule score. What is lost: KL-minimality identity. What is kept: minimal-deviation preference — exactly what curation consumes. **Gateable fingerprint:** mean consumer-score drop of the accepted set vs raw pool ≤ ε; monotone ledger (no accepted move ever decreases the score); 100% provenance audit with an out-of-constraint injection canary.

### 3.3 Fusion — what paper × stack produces that neither has alone

Closest cousins (read, not name-matched):
1. **riir-reflex synth lane** (`src/harness/runner/synth.rs` + `corpus_ab.rs`): proposer (cross-frame span transplant) + C (gold label, shape, dedup, cal-exclusion) + teacher veto. **Lacks the core leg**: acceptance is teacher-agreement (`veto_accept`), single-pass, no score that rises. Fusion = add the I-projection ascent leg: candidate admitted iff teacher-veto holds AND frozen-encoder density does not degrade (with minimal-deviation + monotone ledger); the cohort-level corpus-ab V5 + OOD rig stay as the outer gates. → **Issue 064**.
2. **katgpt-rs `skill_opt::ValidationGate`** (+ `riir-games/src/skill_opt/bomber_skill_opt.rs`): the accept-iff-score-rises skeleton, shipped — but benchmark-scored, skill-doc-scoped, no Metropolis temperature, no constraint-provenance discipline. Signal-diff confirms a real gap (task score vs density). The generic operator (scorer trait + proposer + provenance audit) should be extracted into shared substrate **only when a second consumer lands** — the reflex/training consumers come first (DRY; no parallel substrate).
3. **riir-refine `frontier_miner`**: iterative propose→verify→commit at the capability frontier — oracle-scored by design. Discard reason in §2 row 6.
4. **neuron-db `free_energy_ledger`**: the only shipped Δ-based energy-accept rule (over memory merges). Discard reason in §2 row 7.

Novel combination on record: **projection-sampled SFT corpora for the stack's five training lanes** (Plan 438) + **learner-density ascent in corpus synthesis** (Issue 064). Neither exists in the workspace; the paper is the technique's prior art (we distill it, not claim it).

### 3.4 Privileged-proposer capability import

The 0% → 67.2% result is the strongest modelless fact: the gain was **imported from context** (the expert solution in the proposer's window) and merely *certified* by the unprivileged scorer — capability transfer at curation time, zero gradient. Any stack surface with a privilege slot (answer key, oracle trace, retrieved exemplar, spec) can run the recipe.

---

## 4. Adversarial panel outcomes (§3.5, one spawn round)

- **No-GD advocate**: 17-row inventory; verdict — the pipeline is 100% gradient-free; top extractions = universal curation operator, proxy-I-projection with measurable degradation, privileged-proposer import. Greedy-vs-MH answered: greedy is right for curation; diversity from the walk ensemble; tempered accept `σ(Δ/T)` is the drop-in upgrade. **Adopted** into §3.
- **Model-based advocate**: full recipe table + five-pipeline ranking. **TOP consumer = civ DualLeoTrainer BC leg** (~zero GPU; deterministic replay + exact payoff = a strictly stronger verifier than the paper's MCQ keys; off-policy traces are the paper's exact patient; documented convergence history). Runner-up: quest-grammar as the sub-GPU-hour calibration pilot. Also priced: RIDT v2 distill upgrade (echo-chamber risk → OOD gate), NLEH heads (frozen-read budget), Bonsai .bits (27B scoring = 2–120 h — flagship-only). **Adopted** into Plan 438.
- **Discards**: §2 rows 6–7 (healer density term; consolidation accept) — reasons recorded per the mechanism-level scrutiny rule.

## 5. Prior-art landscape (§4 web search)

- **Not anticipated** (paper is 1 day old; zero citations; treated as strong prior, not proof). No published work combines MH chain + I-projection-of-base-onto-information-constraint + consumed-as-plain-SFT-data.
- **Closest class**: sample-then-SFT filtered self-training (RAFT, ReST/ReST-EM, RFT, BoNBoN/InfAlign) — all accept by *external quality* from *self-generated* samples (sharpening). The paper's delta: likelihood-acceptance polarity (learnable vs quality), off-policy expert source (beyond-sharpening), sequential MH projection vs one-shot filtering, learner-as-reference geometry.
- Adjacent: GKD/Qwen3-OPD/OPSD (objective-level, teacher-side); twisted SMC/TWIST + Variational-BoN (projection-sampling machinery, inference-time); Kumar et al. constrained MCMC (generation, not data); DART (robotics perturbation ancestor); instruction-backtranslation (quality-scored rewrites, no projection semantics). Watch item: twisted-SMC groups porting their machinery to data shaping.

## 6. Verdict

**Tiers:** no Super-GOAT (the modelless consumers are internal-quality surfaces — selling-point test fails; the training track is recipe adoption of published work — GOAT-tier, not a moat-owned novel fusion). **Gain overall.**
- **Training track: GOAT** — direct, measured-gain plan across five lanes with kill-gated pilots. → riir-train Plan 438.
- **Modelless track: Gain (fusion ideas)** — corpus-synthesis ascent leg → riir-reflex Issue 064; two audited discards recorded.

**MOAT gate (§1.6):** riir-train row — "training-method implementations + configs + trained weight assets": the plan fits squarely. Note lives here (katgpt-rs) per house convention (346/524 precedent); the work files in the owning repos. Fusion-priority ladder check ("what does this do for the healer?"): asked and answered — discard recorded (§2 row 6).

## 7. Files

- Plan: `../../riir-train/.plans/438_projection_sampling_sft_data_shaping.md`
- Issue: `../../riir-reflex/.issues/064_projection_ascent_corpus_synthesis.md`
- 4090 queue: `../../riir-train/.issues/610_projection_sampling_scoring_queue.md`
- Code provenance: cloned to `.raw/finetuning-with-sampling` `@ 6d3e9f0b`, deleted after distill.

---

> **PASS-Redirects (synthesis):** Blondel et al. [arXiv:2611.16736 "On the Difficulty of Sequence-Level Supervised Learning for Autoregressive Language Models"] (2026-11-12) — the theory twin of this note's remedy axis: intrinsic decoding-aware sequence-level LOSSES are unbounded below (greedy + sampling forms), so "change the data distribution, never the loss" (this note's projection-sampling SFT shaping; the paper's own §6 recommendation) is the proven-correct lever; also pins the scheduled-sampling inconsistency (sampled prefix + ground-truth target) that plan 501's design already avoids by re-querying the teacher under the perturbed prefix. Sweep lesson recorded 2026-11-18: two live mixed-provenance sites (`dflash2_chat_head_train --onpolicy-frac`, `dspine_distill onpolicy_frac`/MIXED-FRAC — both draft-correction objectives, sound as conditionals) were missed by an exposure-bias-vocabulary grep; mine by knob spelling (`onpolicy|frac`), not mechanism name.
