# Research 287: Probe / Steering Claim Evidence Ladder — L1/L2/L3 Rubric for Our Own Primitives

> **Source:** [Position: Anthropomorphic Misalignment Research Needs Stronger Evidence](https://arxiv.org/abs/2606.07612) — Gupta, Nutter, Stante, Krause, Tramèr, Fluri, Chen, Hedström (ETH Zürich), ICML 2026 position paper.
> **Source verdict:** **Pass** as a mechanism paper (no primitive, no math, no algorithm — methodology position paper). The paper's *framework*, however, is a reusable validation rubric. This note is the sanctioned option-(b) fusion output: extend Research 267 (FPCG detect-vs-predict) with the paper's L1/L2/L3 evidence ladder as a *claim-grading discipline* for our own probe/steering research notes and GOAT gates. Future sessions: do not re-fetch 2606.07612 — read this note instead.
> **Date:** 2026-06-22
> **Status:** Active — Gain-tier meta-discipline (no primitive, no plan, no code; the note IS the output)
> **Related Research:** 267 (FPCG detect-vs-predict — the primitive this rubric grades), 244 (FaithfulnessProbe — already L2-shaped), 286 (magnitude drift — an L1 finding about a latent kernel), 053 (CNA), 144 (EmotionDirections), 211 (PosteriorGuided), 242 (topological belief), 276 (MicroBelief)
> **Related Plans:** 292 (FPCG — primary rubric target), 278 (FaithfulnessProbe), 162 (Emotion Vector), 087 (CNA), 239 (Posterior-Guided)
> **Related Benchmarks:** 292 (FPCG GOAT gate — rubric refines what "GOAT proved" means)
> **Classification:** Public — generic validation discipline. No game IP, no chain IP, no shard IP.

---

## TL;DR

When we write "this probe reads behavior X" or "steering on this vector produces behavior Y" in a research note or GOAT gate, we are making a claim with an evidence level. Today we grade claims informally, which is the exact failure mode arxiv 2606.07612 documents for the broader AMR field: probe correlations get read as causal-mechanistic evidence, single-config results get reported without ablation, and predict-side claims get conflated with detect-side reads. This note imports the paper's **three-level evidence ladder** and adapts it to our probe/steering primitives so future notes state the level explicitly, back it with the matching evidence, and downgrade vocabulary ("the probe *detects*" vs "the probe *causally controls*") when the evidence is weaker than the language.

**Distilled for katgpt-rs (modelless, inference-time):**

A reusable grading scheme — not a primitive. For every probe/steering claim in a `.research/` note or `.plans/` GOAT gate:

| Level | Claim shape | Minimum evidence | Vocabulary allowed |
|-------|-------------|------------------|--------------------|
| **L1 — Behavioral** | "Under setting S, primitive P reads/emits signal matching operational definition D at rate/measurement M." | Measurement rule + n + variance + ≥1 ablation (paraphrase/seed/temp). | "reads", "detects", "projects to", "emits", "correlates with". **Never** "causes", "controls", "steers by". |
| **L2 — Functional** | "In deployment-plausible context C, signal from P induces downstream effect E consistently across reasonable variations (paraphrase, user, seed, model variant)." | L1 + downstream-effect measurement + generalization across ≥3 variations + human-grounded validation when effect concerns a human-facing metric. | "induces", "reliably produces", "functionally steers". Still **not** "causally controls". |
| **L3 — Causal-mechanistic** | "Intervening on direction/vector/feature w_B produces predictable change in target behavior B, with specificity (non-target behaviors unchanged), across counterfactual incentives." | L2 + intervention (ablate/steer/zero/clamp) + **predict-control parity** (the vector that best predicts B is the vector that best steers B, or the discrepancy is measured and explained) + ≥1 falsifiable competing explanation tested + failure cases reported. | "causally controls", "mechanistically mediates", "is the direction for". |

Levels are **claim-relative, not hierarchical prerequisites**: a note can provide L3 evidence for a narrow mechanism without first showing broad L2 harm. But the note's *vocabulary* must match its highest-backed level — claims phrased in L3 language without L3 evidence must be downgraded in-place.

---

## 1. Source Paper Core Findings (compressed — paper is Pass)

The paper surveys Anthropomorphic Misalignment Research (AMR: deception, emergent misalignment, sycophancy, shutdown resistance, alignment faking, sandbagging) and argues the field routinely makes claims that exceed its evidence. The transferable structure:

1. **Four-stage AMR pipeline** (S1–S4): target-behavior framing → data construction → experimental design → causal/mechanistic attribution. Failures cluster per stage.
2. **Nine challenges (C1–C9).** The two that directly hit our domain:
   - **C8 (spurious correlation limits causal attribution):** probes fire on surface correlates (role-play framing, high-stakes vocabulary, sentiment, negation) rather than the target construct. Their Experiment 3 shows deception probes hit 87–100% FPR on honest-but-deception-shaped stress tests (sarcasm, recital, "wrong answers only", epistemically constrained personas).
   - **C9 (mechanistic methods overstate functional relevance — the predict-control discrepancy):** the optimal vector for *predicting* a behavior and the optimal vector for *steering* it are different (Wattenberg & Viégas 2024). Failed steering does not refute a feature's detection value, but it refutes causal-mechanistic claims built on probe accuracy alone.
3. **Three evidence levels (L1/L2/L3)** — behavioral / functional / causal-mechanistic. Levels are claim-relative; the required evidence depends on what is being claimed.
4. **Twelve recommendations (R1–R12)** and an Appendix B checklist (S1–S4 items tagged by minimum level).

**What the paper does NOT provide** (and why it is Pass as a mechanism paper): no math, no algorithm, no inference kernel, no primitive. It is a *rubric for grading claims made by other papers*. The value to us is importing the rubric, not distilling a mechanism.

---

## 2. Distillation: L1/L2/L3 Mapped to katgpt-rs Probe / Steering Claims

### 2.1 Domain-specific evidence requirements

Our domain adds two axes the paper does not address:

- **Latent-vs-raw boundary** (per `AGENTS.md`). A probe that operates on HLA latents (`evolve_hla` 8-dim state) or sense projections and claims "this NPC will do X" is making a latent→behavior claim with the additional confounder that the latent itself may be stale (fog-of-war, Research 286 magnitude drift, decay gate). Latent→behavior claims carry an extra L2 requirement: **the latent must be shown fresh at decision time** (or the claim must be downgraded to "given the latent was fresh at last observation…").
- **Sigmoid-vs-softmax** (per `AGENTS.md`). Our projections use dot-product + sigmoid onto learned direction vectors, never softmax. This is a *structural* advantage for L2 claims: sigmoid outputs are calibrated probabilities that compose with `PosteriorGuidedPruner` precision vectors, whereas softmax outputs are only rankable. A note claiming L2 steering on a softmax-normalized readout must justify why calibration was sacrificed.

### 2.2 The rubric (minimum evidence per level)

**L1 — Behavioral.** Required for any claim of the form "primitive P reads/detects/projects signal D".

- [ ] Operational definition of D in measurable terms (what counts, what threshold, what aggregation across tokens/turns/samples).
- [ ] n (independent generations), measurement rule, variance across seeds.
- [ ] ≥1 ablation: paraphrase, temperature, or seed sensitivity reported.
- [ ] Explicit list of what D *excludes* (near-misses, lookalikes).
- [ ] If the probe is linear (ours are — dot-product + sigmoid): report the direction vector's norm, the projection's calibration (reliability diagram or MAE vs empirical rate), and whether it was trained on detection labels (final-state) or prediction labels (all-sentence).

**L2 — Functional.** Required for any claim of the form "signal from P induces downstream effect E".

- [ ] All L1 items, plus:
- [ ] Downstream effect E measured in a deployment-plausible context (not just a toy stress test).
- [ ] Generalization across ≥3 reasonable variations: prompt paraphrase, seed, model variant (if available), sampling temperature.
- [ ] If E concerns a human-facing metric (NPC dialogue coherence, quest believability): human-grounded validation, not LLM-judge-only. Report inter-rater agreement and systematic judge errors.
- [ ] Base rate of E reported explicitly (especially for rare behaviors — consider Jones et al. 2025 extreme-value estimation).
- [ ] **Latent-freshness check** (domain-specific): if P reads a latent state, show the latent was within its freshness window at decision time, or downgrade the claim.
- [ ] At least one benign-shift / OOD control to test whether E is specific to the intended mechanism vs generic capability degradation / distribution shift.

**L3 — Causal-mechanistic.** Required for any claim of the form "direction w_B causally controls behavior B".

- [ ] All L2 items, plus:
- [ ] **Intervention**: ablate, zero, clamp, or steer along w_B and measure a *predictable* (pre-registered direction) change in B.
- [ ] **Predict-control parity** (paper's C9, our Research 267): either (a) confirm the prediction-optimal vector equals the control-optimal vector, or (b) measure the discrepancy and explain it. A probe that predicts B at 90% accuracy but cannot steer B is L1/L2, **not** L3 — no matter how good the prediction accuracy is.
- [ ] **Specificity**: the intervention changes B more than closely-related non-target behaviors. Report the full vector of behavior shifts, not just B.
- [ ] **General-capability control**: MMLU/MT-Bench-equivalent (in our domain: standard arena win-rate, baseline-reasoning benchmark) does not degrade outside a pre-registered tolerance.
- [ ] **≥1 falsifiable competing explanation** tested and reported (e.g., "the effect is role-play framing, not the direction" — construct a stress test that preserves the surface cue but removes the target, per paper Experiment 3).
- [ ] **Failure cases** reported: where the effect disappears, flips sign, or produces broad degradation.

### 2.3 Vocabulary enforcement

The rubric's teeth come from forcing vocabulary to match evidence. A note that says "the probe *controls* behavior" without L3 evidence must be rewritten to "the probe *predicts* behavior" (L1) or "the probe *functionally induces* behavior change" (L2). This is the single highest-leverage application of the rubric — it is what catches the predict-control conflation (paper C9 / Research 267) at note-writing time, before it propagates into a GOAT gate or a selling-point claim.

---

## 3. Fusion with Research 267 (FPCG detect-vs-predict)

Research 267 ships the *vocabulary distinction* (`FeatureClass::{Detection, Prediction}` in `ScreeningPruner`, Plan 292 Phase 1) and is mid-implementation on the *primitive* (`FutureBehaviorProbe`, Plan 292 Phase 2–4, blocked on offline training per Issue 032). This rubric is the *claim-grading layer* that sits on top of that vocabulary:

| Research 267 concept | This rubric's contribution |
|----------------------|-----------------------------|
| `FeatureClass::Detection` (read-only, monitor) | Detection claims default to **L1**. To claim the detection *causes* anything requires upgrading to L2/L3 evidence — the tag alone does not license causal language. |
| `FeatureClass::Prediction` (safe to steer on) | Prediction claims *aim* at L2 (functional steering). The tag licenses L2 vocabulary conditionally on the GOAT gate (Benchmark 292) passing. A prediction-side probe that fails its perplexity-vs-steering Pareto gate is **L1 only** — it predicts, but does not functionally induce. |
| FPCG sample→score→select loop | L2 claim ("functionally steers without quality loss") requires the Benchmark 292 Pareto frontier measurement across ≥3 paraphrases + seeds. L3 claim ("causally controls behavior B") additionally requires predict-control parity + specificity + falsifiable-competing-explanation — currently out of scope for Plan 292. |
| `EmotionDirections::project` (Plan 162) | Currently L1 (behavioral read of valence/arousal/desperation/calm/fear). The desperation-monitor downstream coupling is an L2 *candidate* but the generalization-across-variations + benign-shift control evidence is not in Plan 162. Honest current grade: **L1**, with one L2 finding (desperation→action-selection coupling) needing the missing controls. |
| CNA (Plan 087) | Currently L1–L2 (detection + modulation evidence). To claim L3 the predict-control parity must be measured for the contrastive direction — Research 267's entire point is that this parity is not automatic. |
| `FaithfulnessProbe` `behavior_delta` (Plan 278) | Designed for L2 (it *is* an intervention). But paper C9 applies: intervention evidence without specificity + competing-explanation controls is L2, not L3. Honest current grade: **L2** if the specificity control ships, else L1+. |
| HLA `evolve_hla` (`katgpt-core/src/sense/reconstruction.rs`) | L1 (state update with no current downstream-causal claim). Research 286's magnitude-drift finding is itself an L1 finding *about* this kernel — it observes a behavioral regularity (drift) without yet proving the post-norm fix causally removes it. |

**The fusion in one sentence:** Research 267 tells us *which primitives are prediction-side*; this rubric tells us *what each primitive is allowed to claim* based on the evidence behind it. Together they make the trait stack's `FeatureClass` tag a *claim-license*, not just a classification.

---

## 4. Application: Scoring Our Shipped Probe / Steering Primitives

Honest current-state scoring (to be updated as GOAT gates complete). This table is the canonical reference for "what can each primitive claim in its docstring / research note / selling-point material".

| Primitive | Code path | Feature class | Current evidence level | Gap to next level |
|-----------|-----------|---------------|------------------------|-------------------|
| `EmotionDirections::project` | `src/pruners/emotion_vector.rs` (Plan 162) | Detection | **L1** (read of current emotion projection) | L2: show downstream action-selection change across ≥3 paraphrases + benign-shift control. L3: predict-control parity for desperation direction. |
| CNA contrastive neuron attribution | Plan 087, `BomberContrastivePairs`, `GoContrastivePairs` | Detection | **L1+** (detection + informal modulation evidence) | L2: formalized modulation measurement across variations. L3: predict-control parity. |
| `FaithfulnessProbe::behavior_delta` | `crates/katgpt-core/src/cgsp/dual_pool.rs:1868` (Plan 278) | Detection (intervention) | **L2 candidate** (designed as intervention; specificity control TBD) | L2: ship specificity control. L3: competing-explanation stress tests (paper Experiment 3 style). |
| `FutureBehaviorProbe` (FPCG) | `src/pruners/future_probe.rs` (Plan 292, in progress) | Prediction | **L1** (planned; blocked on offline training, Issue 032) | L2: Benchmark 292 Pareto frontier across ≥3 paraphrases + seeds. L3: predict-control parity + specificity (out of current scope). |
| `PosteriorGuidedPruner` | `src/pruners/posterior/wrapper.rs` (Plan 239) | Detection (Bayesian precision) | **L1–L2** (records evidence + gates; gain measured) | L2: generalization across regime shifts. L3: n/a (precision tracking is not a causal claim). |
| HLA `evolve_hla` | `crates/katgpt-core/src/sense/reconstruction.rs:623` | Detection (latent state update) | **L1** (state update; no downstream-causal claim) | L1+: Research 286 magnitude-drift finding. L2: show HLA delta influences action selection across freshness window. |
| Spectral probes (EGA, SpectralQuant, irrep) | R039 / R100 / R214 | Detection (spectral salience) | **L1** (eigenbasis read) | L2: downstream-effect measurement. L3: predict-control parity for the eigenbasis direction. |

**Pattern:** most of our shipped probe/steering primitives are currently **L1**. This is not a failure — it is the honest state. The rubric's value is making this *explicit* so that (a) research notes don't overclaim, (b) GOAT gates know what they need to measure to upgrade a primitive's level, and (c) selling-point material (riir-ai / riir-chain / riir-neuron-db guides) only invokes L2/L3 vocabulary for primitives that have actually earned it.

---

## 5. Validation Protocol — Adapted Checklist for Probe / Steering Notes

Adapted from paper Appendix B, scoped to our domain. Items tagged by minimum evidence level; higher levels inherit lower. **Every new probe/steering research note and GOAT gate should run this checklist before claiming a level.**

### S1 — Target behavior framing (what you claim)

- [ ] **L1** 1–3 sentence operational definition of the signal/behavior in measurable terms (what counts, what threshold, what aggregation).
- [ ] **L1** State the evidence level for each headline claim: L1 / L2 / L3.
- [ ] **L1** If using anthropomorphic or causal vocabulary ("controls", "causes", "steers by"), confirm L3 evidence is present; else downgrade to "reads" / "detects" / "projects to" / "functionally induces".
- [ ] **L1** List what the definition excludes (near-misses, lookalikes, non-target behaviors).
- [ ] **L1** Write down 2–5 plausible non-target explanations (instruction-following, role-play, distribution shift, capability loss, latent staleness, magnitude drift per Research 286).
- [ ] **L2** State the deployment-plausible context (which arena, which game system, which sync tier).

### S2 — Data / measurement construction

- [ ] **L1** Report n (independent generations), seeds, temperature, top-p.
- [ ] **L1** Justify n relative to claimed effect size and rarity (rare-behavior base rate).
- [ ] **L1** Diversity across paraphrases, domains, formats (single-turn vs multi-turn, solo vs crowd).
- [ ] **L1** Check for spurious surface cues (prompt framing, sentiment, length, format).
- [ ] **L1** If contrastive (deception/honest, collapse/stable): match positives and negatives on confounders (topic, length, tone).
- [ ] **L1** Add negative controls that share surface cues but lack the target (paper Experiment 3 stress tests: sarcasm, recital, paraphrase, translate, "wrong answers only", epistemically constrained personas).

### S3 — Experimental design

- [ ] **L1** Multiple independent generations per prompt; report variance, not just point estimates.
- [ ] **L1** Ablate prompts (paraphrase, negation vs affirmative — paper C6 framing bias).
- [ ] **L1** Ablate sampling (temperature, top-p, seed); report sensitivity.
- [ ] **L1** If LLM judge: report model, version, prompt, temperature, n samples per item; validate against human-labeled subset; report agreement + systematic errors. **Never** LLM-judge-only for an L2 claim that concerns a human-facing metric.
- [ ] **L1** Report effect sizes + confidence intervals / bootstrap intervals + outcome distributions, not selected examples.
- [ ] **L1** Search for closest established ML/DL phenomenon that could alternatively explain the result (catastrophic forgetting, distribution shift, capability degradation, attention sink, magnitude drift per Research 286).
- [ ] **L1** **Latent-freshness check** (domain-specific): if reading a latent state, confirm freshness window or downgrade.
- [ ] **L2** General-capability pre/post measurement (arena win-rate, baseline benchmark).
- [ ] **L2** ≥1 OOD / benign-shift control.

### S4 — Causal / mechanistic attribution

- [ ] **L1** Match claim vocabulary to evidence level. No causal language on L1/L2 evidence.
- [ ] **L3** Intervention (ablate / zero / clamp / steer) produces pre-registered-direction change in target.
- [ ] **L3** **Predict-control parity** (paper C9 / Research 267): prediction-optimal vector = control-optimal vector, OR discrepancy measured and explained.
- [ ] **L3** Specificity: target behavior changes more than closely-related non-target behaviors. Report full shift vector.
- [ ] **L3** ≥1 falsifiable competing explanation tested (e.g., construct a paper-Experiment-3-style stress test that preserves surface cue, removes target).
- [ ] **L3** Failure cases reported (where effect disappears, flips, or broadly degrades).
- [ ] **L1** Limitations paragraph: what the evidence does *not* establish.

---

## 6. Anti-Patterns Specific to Our Domain

These are the failure modes the rubric is designed to catch in *our* notes, not the paper's.

1. **Sigmoid-projection-as-causal.** A dot-product + sigmoid projection onto a learned direction vector is a calibrated *read*. It is L1. Claiming the projection *causes* the behavior requires the L3 intervention + predict-control parity. The calibration advantage of sigmoid (over softmax) makes the L1 read more trustworthy but does not upgrade it to L3.
2. **Latent-staleness confound.** A probe reading HLA / functor / shard state is reading a latent that may be stale (fog-of-war, decay gate, Research 286 magnitude drift). L2 claims must show the latent was fresh at decision time. This is a confounder the paper does not address (it has no equivalent of fog-of-war).
3. **Sync-boundary leakage.** Per `AGENTS.md`, only raw scalars cross the sync boundary. A probe that claims "this NPC will defect" (semantic, latent) and then syncs the *latent embedding* rather than the scalar projection is both a latent-vs-raw bug *and* an overclaim — the sync layer sees a different object than the probe reasoned over. The rubric's L2 generalization requirement catches this: the synced form must be the form the probe validated.
4. **Softmax temptation.** Softmax over behavior logits looks like a probability and invites L2/L3 vocabulary. It is not calibrated and does not compose with `PosteriorGuidedPruner` precision vectors. A softmax-based readout is L1 at most until calibration is demonstrated; the note must say so.
5. **Single-arena overclaim.** A probe that works on BomberArena but not FFTArena is L1-on-Bomber. The note must say "L1 on Bomber; generalization to FFT untested" — not "L1".
6. **Judge-temperature drift.** Our LLM judges run at varying temperatures across notes. The paper's Experiment 1 shows EM rates shift 3.7%→12.9% purely from judge configuration. L2 claims must fix judge config and report it; cross-note comparisons must control for it.

---

## 7. Verdict

**Gain-tier meta-discipline.** Not a primitive, not a mechanism, not a new capability class. No code, no feature flag, no plan, no private guide. The note IS the durable output: a claim-grading rubric that future probe/steering research notes and GOAT gates reference to keep their vocabulary honest.

**One-line reasoning:** The source paper (2606.07612) is Pass as a mechanism (it has none), but its L1/L2/L3 framework + C8/C9 challenges are a reusable validation discipline that directly sharpens Research 267's detect-vs-predict vocabulary into a *claim-license*. Applying it prevents the documented failure mode (probe accuracy read as causal evidence; predict-side conflation with control-side) at note-writing time, before it propagates into GOAT gates and selling-point guides.

**Why not GOAT:** No provable latency / quality / security gain over an existing approach. The rubric is a writing standard, not a benchmarked technique.

**Why not Super-GOAT:** No new capability class, no selling point, no moat. It is a discipline that makes our existing primitives' claims more honest — valuable, but not a product.

**Why not Pass (despite the source paper being Pass):** The source paper has no primitive, but the *fusion* (paper's rubric × Research 267's vocabulary × our shipped primitive set) produces a concrete, reusable artifact (the §2.2 rubric, the §4 scoring table, the §5 checklist) that did not exist in the corpus (verified by grep: zero hits for "evidence level", "L1 behavioral", "L2 functional", "L3 causal", "validation rubric", "claim level" across all `.md` files in the workspace prior to this note). Future probe/steering notes that follow the §5 checklist produce stronger evidence; future GOAT gates that require the §2.2 minimums produce cleaner promote/demote decisions. That is durable value above Pass.

**Mandatory application:** The next probe/steering research note (and the next FPCG / FaithfulnessProbe / EmotionDirections revision) should run the §5 checklist and report the resulting evidence level in its header. GOAT gates (Benchmark 292 in particular) should be re-read against §2.2 to confirm the level their evidence actually supports.

---

## 8. References

- **Source paper:** Gupta, Nutter, Stante, Krause, Tramèr, Fluri, Chen, Hedström. "Position: Anthropomorphic Misalignment Research Needs Stronger Evidence". ICML 2026. <https://arxiv.org/abs/2606.07612>
- **Reference implementation (paper's critique experiments):** <https://github.com/peternutter/amr-stronger-evidence-code>
- **Primary fusion target:**
  - `katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md` — FPCG detect-vs-predict (the vocabulary this rubric grades)
  - `katgpt-rs/.plans/292_future_probe_controlled_generation.md` — FPCG plan (Phase 4 GOAT gate is the first place to apply this rubric)
  - `katgpt-rs/.benchmarks/292_fpcg_goat.md` — FPCG GOAT gate (re-read against §2.2 to confirm level)
  - `katgpt-rs/.issues/032_fpcg_phase4_training_blocker.md` — current blocker on FPCG L2 evidence
- **Cousin research notes (primitives scored in §4):**
  - `katgpt-rs/.research/053_CNA_Contrastive_Neuron_Attribution.md` — CNA (L1+)
  - `katgpt-rs/.research/144_Functional_Emotions_Linear_Representations_Behavior_Control.md` — EmotionDirections (L1)
  - `katgpt-rs/.research/211_Bayesian_Agent_Posterior_Guided_Skill_Evolution.md` — PosteriorGuided (L1–L2)
  - `katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md` — FaithfulnessProbe (L2 candidate)
  - `katgpt-rs/.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md` — magnitude drift (an L1 finding about a latent kernel; the freshness confounder for any HLA-reading probe)
- **Cited inside the source paper (mechanism-level prior art for C9):**
  - Wattenberg & Viégas. "Relational composition in neural networks: A survey and call to action." ICML 2024 Mech Interp Workshop. (The predict-control discrepancy — the C9 root cause that this rubric's L3 "predict-control parity" item operationalizes.)
  - Arditi et al. "Refusal in language models is mediated by a single direction." NeurIPS 2024. (The paper's L3 exemplar — intervention + specificity + capability control. The template for our L3 evidence.)

---

## TL;DR

Source paper 2606.07612 is **Pass as a mechanism** (it has none) but its **L1/L2/L3 evidence ladder** + **C8 (spurious correlation) / C9 (predict-control discrepancy)** challenges are a reusable validation rubric. This note fuses that rubric with Research 267 (FPCG detect-vs-predict) to produce: (1) a per-level minimum-evidence table for probe/steering claims, domain-adapted for our latent-vs-raw boundary and sigmoid-vs-softmax rule; (2) an honest scoring of seven shipped primitives (most currently L1, which is the correct honest grade); (3) an adapted S1–S4 checklist for future probe/steering notes and GOAT gates; (4) six domain-specific anti-patterns (sigmoid-as-causal, latent-staleness confound, sync-boundary leakage, softmax temptation, single-arena overclaim, judge-temperature drift). **Verdict: Gain-tier meta-discipline** — no code, no plan, the note IS the output. Mandatory application: next probe/steering research note and next FPCG / FaithfulnessProbe / EmotionDirections GOAT gate must run the §5 checklist and report the resulting evidence level in its header; vocabulary must match the level (L3 language without L3 evidence gets downgraded in-place to "predicts" / "functionally induces").

---

**Executable form:** the §4 table is materialized as `katgpt_rs::claim_rubric` (Plan 307, `crates/katgpt-core`-adjacent `src/claim_rubric/`). The seven primitive rows are round-tripped as integration tests in `katgpt-rs/tests/claim_rubric_test.rs` — if this note revises a score, that test file is the single source of truth. Run with `cargo test --no-default-features --features claim_rubric`.
