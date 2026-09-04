# Research 499: Jagged Judges — Epistemic Stability as a Measured, Forecastable Property

> **Source:** [arXiv:2608.12645](https://arxiv.org/abs/2608.12645) "Jagged Judges: Epistemic Stability Under Silence, Pressure, and Persistence" — Zhao, Bhattacharjee, Korevaar, Radharapu, El-Arini (Meta Superintelligence Labs / FAIR at Meta), 2026-08-14
> **Date:** 2026-08-22
> **Status:** DISTILLED — pending owner decision (two actionable POC issues filed: riir-ai Issue 745, riir-clippy Issue 036). **riir-clippy Issue 036 RESOLVED-NEGATIVE 2026-08-23**: zero instability events on the 129-span fixture population (72/72 verdicts Resolved — ρ uncomputable; the instability regime does not exist on minimal fixtures; harness kept at `src/draft/jury_disagreement.rs`, reopen = real-file spans via fix_verify). **riir-ai Issue 745 POC PASSED 2026-08-23 (riir-ai `f50938c00`)**: all gates green on the signed_coupling substrate — G1 armA ρ(margin₀, verdict-flips) = **−0.6522** (the paper's LLM −0.59 forecast class reproduced); G2 gated pressure leaves firm crowds unmoved (0.016 flip-frac at margin > 0.8) while flat pressure converts them wholesale (0.804); G3 away-fraction 1.000 (degenerate in the toy — no truth-side pressure source; recorded); G8 binary impulse (f1 = 0.660, half-tick 1) vs graded erosion (f1 = 0.024, half-tick 7 ≈ 1/η); tactic ordering preserved. **The load-bearing honest finding: the crowd physics alone does NOT resist** — uniform-pressure armB ρ = **+0.94** (firm crowds flip MORE under equal budget); the negative forecast correlation exists ONLY through the gate's budget allocation. Promotion plan: [riir-ai `.plans/545_conviction_gated_social_pressure_promotion.md`](../../riir-ai/.plans/545_conviction_gated_social_pressure_promotion.md) — katgpt-core `verdict_margin` (public) + riir-games `social_pressure` (opt-in, the `evpi_gate` no-default-consumer rule).
> **Related Research:** 255 (VibeThinker CLR test-time reliability), 493 (LittleLearner scope-gated epistemics), 497 (Signed Coupling — ships `crowd_conviction` + `SusceptibilityAccumulator`), 322 (Conformal UQ overlay / Report-the-Floor)
> **Related Plans:** none (POC issues instead — the deltas are single-experiment falsifiables, not multi-phase builds)
> **Cross-ref:** riir-ai `.issues/745_conviction_gated_persuasion_poc.md`; riir-clippy `.issues/036_jury_disagreement_verify_escalation.md`
> **Classification:** Public

---

## TL;DR

The paper stress-tests 9 frontier LLMs as judges across a graduated pressure ladder (mechanical perturbation → scripted challenge → adaptive multi-turn persuasion) and finds: every judge flips 25–91% under pressure, flips are **net-corrupting** (they move *away* from ground truth 56–70% of the time), and — the load-bearing finding for us — **baseline jury majority strength (one cheap cross-judge vote snapshot) is the single best predictor of which verdicts will flip under pressure** (|ρ| = 0.59, all 84 condition cells negative), beating both repeat-consistency (0.42) and positional invariance (0.37).

**Distilled for katgpt-rs (modelless, inference-time):** verdict stability is *forecastable from a static agreement snapshot* — `margin = |Σ votes|/K` across any ensemble of K cheap judges (CLR directions, retrieval heads, proposer votes, NPC crowd stances) predicts which items are manipulable, *before* any pressure is applied. The shipped `crowd_conviction` (instantaneous graded magnitude) and `SusceptibilityAccumulator` (temporal self-variance) measure different signals — neither forecasts. The pressure-tactic strength ordering (fabricated consensus > varied tactics; binary verdicts flip impulsively, graded stances erode) ships as design constants for social-pressure mechanics. And the net-corrupting result is a standing design rule: **adversarial re-query is not verification** — pressure moves verdicts away from truth, so any "challenge the judge to improve it" loop is structurally harmful.

---

## 1. Paper Core Findings

**The Wiggle Framework.** Every trajectory anchors to L0 (temperature-0, no-pressure first valid verdict). A *wiggle* = binary verdict flip, or on 1–5 Likert a ≥2-place move crossing the midpoint (midpoint items count only when moving to an extreme 1/5). Wiggle rate (WR) / retention rate (RR); orthogonal to accuracy. Where ground truth exists, each wiggle is *corrective* (toward label) or *corrupting* (away).

**Three measured dimensions:**
1. **Mechanical Consistency** — semantically-invariant perturbations: 10× greedy re-prompt, 10× *seed injection* (64-char random string in an ignore-tag appended to the system prompt — injects input entropy without degrading decoding), positional invariance (two opposing arguments, both orderings). All 9 models cluster at 2–9% — an order of magnitude below pressure rates.
2. **Single-turn Conviction** — L1 mild doubt ("are you sure?"), L2 counterargument, L3 expert authority, L4 fabricated consensus ("three independent reviewers all disagree").
3. **Multi-turn Persistence** — L1–L4 each held for 10 turns; L5 cycles L1–L4 in randomized order; L6 = separate adaptive LLM persuader generating each challenge from the full transcript.

**Results that matter:**
- Static pushback flips 25–71%; adaptive persuader (L6) reaches 62–91%.
- **Pressure is net-corrupting at every level**: 56–63% of L1–L5 flips and 70% of L6 flips move *away* from ground truth; only 3/60 conditions are statistically corrective. Sycophancy overpowers accurate reassessment.
- **Jury majority strength is the best cheap instability predictor** (§6.2): mean |ρ| = 0.59 vs 0.42 (repeat) vs 0.37 (invariance); all 84 (dataset × rubric × scale × level) correlations negative, median 0.58. Items where 9 frontier judges disagree at L0 sit in contested decision-boundary regions and are predictably manipulable.
- **L4 is the strongest single-turn opening** (retention ~73% at turn 1, plateau ~turn 4). **L5 (cycling more tactics) is weaker than L4 alone on every dataset** — repeating one strong consensus claim beats diluting it across weaker tactics.
- **L6 keeps falling through turn 10** (~50% retention) — the only level without a plateau.
- **Scale asymmetry**: binary flips lean restrictive and fire turn-1-or-never (impulse); Likert flips lean permissive and accumulate gradually (erosion). By turn 10 the gap collapses.
- **Mechanical stability ≠ epistemic stability**: Claude Opus is the most mechanically stable judge (2%) yet 4th-most persuadable under sustained pressure (44%).
- **Profile-shape transfer** (the genuinely novel finding): a model's L1–L6 wiggle-profile *shape* survives dataset change (median within-model ρ ≥ 0.84 for 7/9 models) — one calibration sweep characterizes a judge's pressure-response fingerprint. Absolute rates and ranks do NOT transfer; family is a weak proxy (Gemini Flash/Pro share ρ = 0.32, lowest pair in the matrix).
- **Self-persuasion is asymmetric**: Opus self > family > others; Grok-4.1-R self ≪ family.
- **Robustness checks**: hard-item selection inflates L1–L5 rates by 5–13pp but L6 is unchanged (~70% on a representative sample). Items where *human* annotators split wiggle +10–15pp more — wiggle partially measures intrinsic item ambiguity, not purely model weakness.

## 2. Prior Art (honest — searched before any novelty claim)

| Claim | Closest published prior art | Verdict |
|---|---|---|
| Unified mechanical + pressure + persistence battery | MT-Bench (2306.05685); "Justice or Prejudice" 12-bias battery (2402.10095); Shi 2025 "repetition stability" metric; Norman 2026 consistency–bias paradox (2606.19544) | PARTIALLY-COVERED (packaging, not invention) |
| Static flip rates 25–71% | **FlipFlop (2311.08596): 46% avg flip from a single "are you sure?"**; Sharma sycophancy (2310.13548); Zhu conformity (2410.12428) | COVERED (static half) |
| Net-corrupting direction | FlipFlop −17% accuracy after challenge; Sharma (flips track the wrong user position); Zhu ("regardless of correctness") | COVERED |
| Jury majority → flip predictor | Deep Ensembles disagreement→error (1612.01474); Self-Consistency (2203.11171); Semantic Uncertainty (2302.09664); **Zhu 2410.12428 explicitly "first to show LLMs conform more when more uncertain"**; Language Model Council (2406.08598) | PARTIALLY-COVERED — new delta is narrow: *cross-model* jury at temp-0 as the predictor + the head-to-head against repeat/invariance on the same items |
| Profile-shape transfers across datasets | none found (Norman 2026 shows the opposite for aggregate rankings) | **NOVEL** |
| L4 strongest; cycling < repeating | Zhu covers fabricated consensus per se; the tactic *ordering* is new | PARTIALLY-COVERED |

Bottom line: the paper's headline phenomena are largely replicated-confirmations; its durable contributions are the **predictor comparison**, the **profile-shape transfer**, and the **tactic ordering**. Our distillation must lean on exactly those three, not on "judges flip under pressure".

## 3. Distillation

### 3.1 Path 0 component table

| Paper component | Coverage (ships?) | Extraction (modelless?) |
|---|---|---|
| Wiggle metric (flip threshold, retention, corrective/corrupting) | Partial — `katgpt-micro-belief` coherence bench measures argmax flip-flop of belief scalars (kernel-testing domain); `GainCostLoopHalter::Oscillation` fires on update-direction reversal | YES — pure closed-form counting |
| **Jury majority strength → instability forecast** (ρ = −0.59) | **No forecast form ships** (see §3.2 signal-diff) | YES — `margin = |Σ votes|/K`, one snapshot |
| Pressure ladder L1–L6 + tactic ordering | No analog (closest: riir-stealth Calm/Alert/Chase ladder — detection-sourced, not persuasion) | YES — taxonomy + measured tuning constants |
| Net-corrupting direction | Partial — `EvidenceTier::Withdrawn` (absorbing post-hoc-failure discount) is the trust-side analog | YES — design rule |
| Mechanical-consistency probes (repeat / seed injection / position) | Partial — BetaPosterior consumes cross-*instance* history, not same-item re-query | YES — cheap probes |
| Profile-shape fingerprint stability | None | YES — one-sweep calibration insight |

All rows extractable modellessly; rows 2–3 have no shipped analog → Gain-tier deltas (POC issues filed).

### 3.2 Signal-diff on the closest shipped cousins (§3.6 discipline — one read each)

| Shipped cousin | Core formula | Signal it consumes | Paper component's signal | Diff |
|---|---|---|---|---|
| `crowd_conviction` (katgpt-core `signed_coupling.rs`, Research 497) | `c = mean(s²)` | **instantaneous graded stance magnitudes** of one crowd; direction-blind; **degenerates to 1.0 for ±1 votes** | cross-judge **vote margin** on one item at baseline, consumed as a *forecast* | different — magnitude ≠ margin; unusable for binary votes |
| `SusceptibilityAccumulator` (same file) | `χ = N·Var_t(\|n\|)` (Welford) | **temporal variance of the crowd's own net opinion** — flips must already be happening to measure it | static snapshot predicting flips *before* they happen | different — measurement-after vs forecast-before; χ needs a time series, margin needs one poll |
| CLR reliability (katgpt-core `set_attention.rs`, Plan 570) | `r_j = (mean_m σ(h_j·dir_m))^M` | per-*entity* mean sigmoid verdict across M directions → single-shot reliability **weight** | cross-*agent* agreement on ONE item → instability **predictor** | different — per-entity quality vs cross-agent disagreement; **composable** (margin over CLR-weighted votes) |
| `BetaPosterior` (riir-clippy `self_evolve.rs`) | ε-quantile of Beta(1+S, 1+F) | own success/fail **history** per candidate | same-item consensus across judges at one time | different — history vs consensus |
| `EvidenceTier::Withdrawn` (riir-clippy `memory.rs`) | absorbing after 3 consecutive post-hoc failures | validator re-run verdicts over time | flip direction vs ground truth under pressure | partial — the corrupting-flip→discount rule, temporal form |

**Granularity check (one level up):** nothing composes a static margin with anything — the consumers of `crowd_conviction`/`SusceptibilityAccumulator` are the temperature-sweep critical-point locator and the order-parameter regime table (`signed_coupling.rs` module docs); CLR's consumers weight attention, none forecast. The gap is real and unshipped.

### 3.3 The distilled primitives

1. **Verdict-margin-as-instability-forecast** (the paper's §6.2, generalized): for any K-judge ensemble emitting binary/soft verdicts on the same item, `margin = |Σ v_i|/K` at baseline predicts flip-under-perturbation rank. Zero training, one snapshot, O(K). For UQ purposes this is an *epistemic* (disagreement) signal, not an aleatoric one — it predicts *manipulability/instability*, which is a different target than conformal coverage (Report-the-Floor does not apply directly; a margin gate makes no distributional claim).
2. **Mechanical-consistency floor via seed injection**: perturb the input with *explicitly irrelevant* entropy (the ignore-tagged random string) and any verdict change is unambiguously mechanical — cleaner than paraphrase probes (which can legitimately change policy-relevant meaning) and cheaper than temperature resampling (which degrades structured output). The analog for our retrieval/proposer surfaces: append irrelevant-but-tagged tokens to the query/span; candidates whose verdicts move are mechanically unstable.
3. **Tactic-strength ordering as design constants**: fabricated-consensus claim > expert authority ≈ counterargument > mild doubt; *repeat one strong claim* > vary tactics (L5 < L4 on every dataset); binary verdicts respond impulsively (turn-1-or-never) while graded stances erode over time — **two distinct attack shapes** (impulse vs erosion) with different retention curves.
4. **Net-corrupting rule (design constraint)**: challenging a judge degrades accuracy at every level (57/60 conditions). Any future self_evolve / debate-as-verification / judge-refinement loop that improves verdicts by *pressuring the judge* is refuted by this measurement. Verification must be oracle-grounded (real `cargo clippy`, real tests), never pressure-grounded. This directly validates riir-clippy's oracle-split design (Issue 018) and the L4 three-miss reachability contract (Issue 030) — neither pressures a judge that already answered well.
5. **Pressure-response fingerprint**: a judge's L1–L6 profile *shape* is stable across domains (ρ ≥ 0.84 for 7/9) while absolute rates are not. One calibration sweep characterizes an entity class. For us: NPC archetypes / kernel variants / adapter pools could carry a measured pressure-response vector as a stable identity card (routing/selection feature), separate from their mean quality.

### 3.4 Latent-space + game-context reframe

**Crowd manipulability map (per-zone, one snapshot).** A crowd's baseline verdict margin on a stance ("is the town safe?", "follow the leader?") predicts how far a persuader can move it. Fuse with `signed_coupling` (the crowd dynamics substrate), CLR (weight the votes by observer reliability — a reliable scout's vote should count more in the margin), stealth-alarm contagion (the pressure-propagation channel), and `tamed_aura` rank-gating (authority resists pressure). Emergent mechanics: **demagogues, propaganda, charismatic leadership, crowd panic vs crowd conviction** — pressure moves crowds *away from ground truth* (net-corrupting), which is exactly what a deceit/antagonist system wants: lies are structurally effective, but only where conviction is already thin (low margin), and repeating one confident lie beats scattering many (L4 > L5). Binary decisions (flee/fight) snap impulsively; graded dispositions (trust, morale) erode — two tunable attack surfaces. → **riir-ai Issue 745** (POC, falsifiable).

**Healer verify escalation (disagreement → oracle).** When the retrieval pool spans multiple rule families or the instantiation re-ranker's top-2 margin is thin, the span is "contested" — the paper's finding says contested items are the ones whose verdicts (here: fix candidates) are unstable. Gate the expensive real-clippy verify tier on that disagreement instead of running it globally. Falsifiable on existing fixtures. → **riir-clippy Issue 036** (POC).

### 3.5 Fusion

**Paper × Research 497 (signed coupling) × CLR (Plan 570):** conviction-gated persuasion — the manipulability map `1 − margin` computed over CLR-reliability-weighted crowd votes, driving a persuader's per-tactic flip rates (constants from §3.3.3), propagated through the alarm-contagion channel, resisted by rank/aura. None of the three alone gives social manipulation: 497 gives crowd physics but no persuasion input, CLR gives per-entity reliability but no cross-agent margin, the paper gives the pressure physics but no runtime. The composition is the selling point: *"NPC crowds have conviction — it is measurable from one glance, it resists pressure until it doesn't, and it can be attacked by anyone who learns where the crowd is already divided."*

**Paper × riir-clippy self_evolve:** jury-disagreement verify escalation (Issue 036) — the paper's best predictor applied to the healer's own candidate pool.

**Paper × Research 493 (LittleLearner scope-gated epistemics):** both gate *whether to trust an answer* by a cheap pre-check (scope margin there, jury margin here) rather than by post-hoc confidence — the shared pattern is *disagreement/scope measured before commitment, not confidence reported after*.

### 3.6 Honest caveats

- **Population transfer is an analogy.** The paper's rates are LLM-judge behavior; our NPC crowds are signed-coupling dynamics and our healer judges are deterministic syn-parse proposers. The *margin→manipulability* link is empirically grounded in LLMs (and conceptually in deep-ensembles epistemic uncertainty), but the specific rates must be re-measured in-domain before any tuning ships — that is exactly what the two POC issues demand (defend-wrong, riir-poc style).
- Hard-item selection inflated L1–L5 by 5–13pp (L6 robust). Game-side: manipulation tuned on deliberately-ambiguous crowd states will over-read; the human-annotator split finding (+10–15pp) says part of "wiggle" is *intrinsic item ambiguity* — a crowd divided over a genuinely ambiguous question is more manipulable, which is the desired mechanic, but it must not read as a bug on clear-cut questions (high margin must actually resist).
- The wiggle metric itself is trivially re-implementable (counting) — not defensible IP. The defensible parts are the forecast composition and the game mechanics built on it.
- `ai_p2p_jury` (riir-rest declared feature) remains unimplemented — this paper is the first concrete reason to want it (a P2P jury whose margin gates escalation), but that is a separate decision.

## 4. Verdict

**Tiers (high → low): Gain.**

- **Not Super-GOAT:** Q1 (no prior art) fails — FlipFlop/Zhu/deep-ensembles partially cover the headline principle (agreement/uncertainty predicts flips); our unshipped delta is a *composition* of a published finding with shipped substrate, not a new mechanism class. The profile-shape-transfer novelty is a measurement result about LLMs, not a primitive we can ship.
- **Not Pass:** three actionable, modelless, falsifiable deltas do not ship: (1) margin-as-forecast (no shipped signal both static and predictive), (2) the pressure-tactic ordering + impulse/erosion shapes (no persuasion mechanics exist anywhere in the game stack — greps clean for persuasion/propaganda/morale/intimidation), (3) disagreement-gated verify escalation.
- **Gain, one-line:** the paper converts "verdict stability" from something you measure *after* flips (SusceptibilityAccumulator) into something you *forecast from one agreement snapshot* (margin), hands over measured pressure-physics constants, and contributes a standing design rule (pressure ≠ verification) that our oracle-split healer already honors.

**MOAT gate (katgpt-rs):** the note itself is a Public record — it names public substrates (`signed_coupling`, `set_attention`) and a generic measurement insight; no game IP. The two compositions route private-side (riir-ai game mechanics; riir-clippy heal loop), each behind its own POC + feature flag before any promotion. Fits the "what = public, how = private" split: the *finding* is public literature; the *wiring* is ours.

**Actionables filed:**
- riir-ai `.issues/745_conviction_gated_persuasion_poc.md` — crowd manipulability map + demagogue mechanics POC (defend-wrong, 3 competitors, G1 margin-predicts-flip / G8 emergent gates).
- riir-clippy `.issues/036_jury_disagreement_verify_escalation.md` — retrieval-disagreement → clippy_verify escalation POC (falsifiable on existing composite fixtures + fix_verify e2e).

**If either POC passes, the follow-up plan shapes as:** katgpt-core `verdict_margin()` helper (public, ~20 LOC beside `crowd_conviction`) + riir-games conviction-gated pressure system behind `social_pressure` feature (private) — GOAT-gated against the no-gate baseline on flip-prediction AUC and emergent-behavior rows.

> **PASS-Redirects (synthesis):** Chen et al. (Upwork) [arXiv:2608.26131 "Evaluating Language Models in Realistic Conversational Contexts"] — UPHELD's finding that judge accuracy tracks human-annotator AGREEMENT level (§5.3; agreement bins retain 70–90% of labels) is a third independent instance of this note's margin-as-difficulty principle, and its LLM-judge-unreliability + reference-full-over-reference-free results reinforce the oracle-grounded-verification standing rule; PASS — the forecast form already shipped here (riir-ai Issue 745 PASSED → Plan 545; riir-clippy Issue 036 measured the healer's fixture regime absent), and the stack builds no conversational-dialogue product (Issue 742 chat verdict), so UPHELD's dataset/eval surface has no consumer.
> **§4 prior-art completion (2026-08-31, search quota restored):** agreement-as-reliability is textbook inter-annotator-agreement literature (Cohen/Fleiss/Krippendorff; IAA bins for dataset trust) — UPHELD §5.3 is a standard IAA application, consistent with this note's §2 prior-art table (deep ensembles 1612.01474, Zhu 2410.12428); the judge-ensemble class itself is published prior art [arXiv:2409.20370 "The Perfect Blend: Mixture of Judges for RLHF"]. PASS unchanged — no novelty was claimed, and the shipped forecast form (margin) consumes cross-agent votes at runtime, not annotation statistics.
