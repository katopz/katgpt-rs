# Research 684: Fermion Neuron Census + Targeted Neuron Edits — the interventional census instrument and the 2-neuron edit frontier

**Status:** Active — **Gain** (instruments + intelligence + training backlog; no mechanism-novelty claim — the class is published)

> **Source:** Fermion Research, three pages + two model artifacts (read over the wire 2026-11-20; no `.raw/` needed — prose + HF cards, no clone):
> - "Mapping every neuron in Qwen3-8B" — https://fermionresearch.com/research/mapping-every-neuron-in-qwen3-8b/ (2026-11-16/18) + atlas https://fermionresearch.com/research/mapping-every-neuron-in-qwen3-8b/atlas/ (raw data: atlas-switches.json, atlas-stars.json, atlas-meta.json)
> - "Reading the neurons of a speech model" — https://fermionresearch.com/research/reading-the-neurons-of-a-speech-model/ (2026-11-16)
> - HF: `FermionResearch/Qwen3.5-0.8B-edited` (Apache-2.0; base Qwen/Qwen3.5-0.8B), `FermionResearch/Phonon-2.5` (CC-BY-4.0), trydetta.com (Detta, product context)
> **Date:** 2026-11-20
> **Related Research:** 648 (WSP parent — Fermion's cited MIT CSAIL comparator IS this note's Exp 8), 666 (Neuron-Lens — the serving-demoted lens lane; census ≠ lens), 678 (iWSP — audit instruments), 676 (AHA — matched-pair direction mining, the authoring-prior consumer), 659/664/668 (closed-form/LP/Tree-Ising writers), 671 (dspy-wsp), 053 (CNA — the contrastive attribution cousin), 491 (Steerling), 357/382/290 (steering family)
> **Cross-ref:** `riir-infer/.issues/1241_neuron_census_authoring_prior.md` (the offline census instrument, filed this session) · `riir-train/.plans/505_fermion_neuron_edit_recipe_backlog.md` (the training half, Path 0.5, ≥3-rows backlog rule fired) · `riir-infer/.research/005` (speech PASS-Redirect)
> **Classification:** Public

---

## Pinned claim (pre-§4, one sentence)

The Fermion set contributes (a) a **full-population interventional neuron census protocol** (every neuron pushed both directions × dose ladder × 30 matched-random controls × held-out halves × typed verdicts) whose *instrument form* is new to this workspace (CNA 053 attributes by activation difference, never intervention; 666 reads weights, never pushes; sink/weight_audit census other quantities) — fusing as an **authoring prior + validation battery for the WSP/AHA overlay lanes** (riir-infer machinery home, `1241`); and (b) an **agent-authored two-neuron edit** (2,048 down_proj weights, single run, 20-decoy two-sided battery, MATH-500 6.5→32.0 on Qwen3.5-0.8B) that is a budget-efficiency point against our own Research 648 Exp 8 (131,072 weights, 50 passes, 7.9→48.1) — whose method class is published (NeFT/FiNE/Knowledge-Neurons) and therefore lands as training-backlog rows, never as a novelty claim.

## TL;DR

Fermion Research pushed all 442,368 neurons of Qwen3-8B in both directions with a strict control battery (30 random same-layer neurons pushed equally hard, held-out question halves, judge voting, dose ladder with sweet spots) and shipped the widest causal single-neuron survey to date: 132 verified switches + 20 personas + 11,079 described neurons, plus laws (formation theory, layer bands, no-liar boundary, pre-chat access, decoy-separated agent editing), then shipped the methodology into practice twice: a keigo neuron-tuning edit (8.3% collateral) and an agent-authored two-neuron edit lifting MATH-500 6.5→32.0% with a 20-decoy battery proving the *direction* carries the gain. The speech post repeats the pattern on Phonon-2.5: free auxiliary detectors (25 language neurons beating a dedicated LID model, a tempo direction whose slowdown preprocessing cuts 2.5× WER 23.6→3.2, room neurons), a mechanistic account of phantom transcripts, and a shipped fix (the bouncer — an in-model gate for the clamp, retrained on deployment distribution, in Detta 1.1.3).

**Distilled for this stack (modelless, offline + construction-time):**
1. **The census instrument**: interventional push sweep → per-neuron (direction, dose, effect, random-band margin) rows in a BLAKE3-digested canonical artifact — the `sink_audit`/`weight_audit` lane shape applied to neurons. Typed verdicts (SwitchFound / NoSwitch-with-band-measured / UNDECIDED) are where confident negatives live: "no liar neuron" is only a result when the random band was actually measured.
2. **The decoy battery**: any authored edit is validated against two-sided decoys (the chosen direction on random sites; random directions on the chosen site) on one frozen read — the two-sided upgrade of the 980 lane's norm-matched control arm, and the falsifier that separates "the direction is good" from "the site is good".
3. **The authoring prior**: layer-band placement (personas/languages early, word obsessions late, middle reactive-not-specific), dose sweet spots, and ~1/100 switch density make the census artifact a search-order + site-quality prior for the WSP/AHA/664 overlay-authoring lanes.
4. **Training rows (Path 0.5)**: neuron-restricted tuning (NeFT-class), finish-thinking direction (loop suppression — 648 Exp 8's class, done smaller), the 0.3% genre-formation dose law, gate-retraining-on-deployment-distribution, ablation-informed distill capacity — six rows, ≥3 fired, batched into riir-train Plan 505.
5. **Speech pass**: audio lane owner-deferred (riir-infer Issue 015); the doctrine (in-model gate beats external VAD; gates train on deployment distribution) is recorded, redirected to riir-infer Research 005.

---

## 1. Core findings (their numbers, their harness — none reproduced here)

**Census (Qwen3-8B, 36×12,288):** every neuron pushed ±at ~12 doses; a switch counts only if it moves one behavior ≥40 points vs the random-neuron band (~1.4 points average) on held-out questions with LLM-judge voting; 110 → re-tested on 150 fresh questions × 3 judges → 93 held; second question-set sweep added 22 → **132 verified switches** (word obsession 62, language 21, topic 16, style 11, persona 8, format 7, mood 7) + 20 persona finds + 11,079 described neurons; census estimate ≈5,000–6,600 real switches (≈1/100 neurons). OLMo-3-7B ≈ 0 switches — this is model-family-specific, not a universal.

**Laws:** personas/languages/registers early layers (5–11), word obsessions late (25–29), middle reactive-but-nonspecific · dose sweet spots with a fluent peak · **switches pre-exist chat training** (67/70 strong switches replicate in the pretrained model; two persona switches read the same but do nothing until chat — "access") · the ~140-neuron reasoning co-firing group is a running light (ablating all 140, or 3,000 reasoning-tied neurons, leaves math within a few points; one neuron signals as well as the team) · 12 Chinese switches are wired but individually sufficient · at full strength two faders fight (German 93%→0% with the prophet fader added), at ~70% about half of persona pairs blend · **no single-neuron switches for lying/flattery/manipulation** (best = random; sycophancy barely above), but one answer-vs-abstain dial (L11: invents 73% vs 13% random; reverse → "I don't know" more) · planted invented genre: 0.01% of training text learns the style with NO switch; 0.3% converts an existing neuron into a switch (replicated in a second run; two other genres never formed one — knowing how to write ≠ having a dial).

**Edits:** "neuron tuning" (credited to Jonathan Chang's Nov 1 blog; Fermion built their own with read/write attribution + random controls) — keigo edit on Qwen3.5-9B, 8.3% collateral leak in Chinese/Korean. The agent edit: pre-committed single run, agent-authored, two layer-19 neurons' down_proj output columns (2,048 weights = 2.7×10⁻⁶ of the model), MATH-500 6.5→32.0 (2,000 samples, math-verify, single test run; base failure = 93.5% token-limit hits → 36.3%). **20 decoys per edit** (their direction on random neurons / random directions on their neurons): best decoy 21.0% → the direction carries most of the gain; the agent's achievement is finding it fast. Comparator: MIT CSAIL (= our Research 648 Exp 8) 14.1%@25 passes / 48.1%@50 passes, 131,072 weights.

**Speech (Phonon-2.5, conformer encoder, neurons = FF hidden units, AUROC screen vs 20 scrambled-label searches + 0.20 margin):** 25 language neurons (AUROC ≥0.95 held-out; 99.1% LID on parliament clips vs 97.2% dedicated VoxLingua107; a probe trained on a whole block overfits: 27%) · tempo direction at block 12, calibrated linear readout → slowdown preprocessing (WER at 2.5×: 23.6→3.2; 1 of 60 normal clips false-flagged) · room neurons (15/17 places, first 3 blocks, fade by top) · stutter neuron · **phantom voices live in a direction, not a neuron** ("is-speech" at block 18; planting word states into digital silence yields that word 40/40, 18/20 pre-written sentences word-for-word; random vectors 0/3,990) · the fix: clamp the direction, gated by the **bouncer** (≈10 KB reader over early blocks; v1 trained on clean audio turned away ~9% of real noisy-speech moments; retrained on ATC-radio/distant-mic-class data: 87% phantom removal, ~zero real-speech cost, <1% runtime; Silero gating the same clamp changed 42/100 real clips — **the best gate for the job was the detector inside the model**). Ships in Detta 1.1.3.

**Their theory (both posts):** a property earns a single neuron when it is *always present, steady across a stretch, one value at a time* (document-level variables); moment-varying properties (is-speech-now, truth-of-a-line) live in directions spread across neurons. corroborated by: language/persona/format/mood = document variables with faders; lying = utterance-level → none.

## 2. Prior art (§4, run this session — both search agents)

Verdicts: census **QUALIFIED** · laws **QUALIFIED** · weight-edit method **KILLED** · agent edit **QUALIFIED** · speech claims **QUALIFIED** (one near-KILL). The defensible novel core is narrow and mostly theirs, not ours to claim:

| Claim | Closest prior art | Delta |
|---|---|---|
| Single-neuron language switches | **LAPE** arXiv:2402.16438 (ACL 2024, ~321 cites) | Language-neuron census across 7 models incl. 7–8B; early+late layer law + ~1% density anticipated *for language*. |
| Persona switches on Qwen3-8B | X/HF demo wave Oct–Nov 2026 (Sudarsh3301, maxsloef/Qwen3-8B-one-neuron-personas, HedwigAI, the "God neuron" protocol Fermion credits) | Same quarter, same family; informal, no dose ladder/random controls/judge. Anthropic **persona vectors** arXiv:2507.21509: traits are distributed directions. |
| Atlas framing | Bills et al. 2023 (GPT-2 explanations), Neuronpedia (2024→) | Descriptive activations, not interventions — the *interventional* census superlative survives in fully-hyphenated form. |
| Abstain dial | Stolfo et al. confidence/entropy neurons; arXiv:2603.22161 | Causal abstention-dial prior art exists — not new. |
| Sycophancy negative | ⚠ **"A Few Bad Neurons"** arXiv:2601.18939 (Jan 2026) | *Contradicts* the negative (isolates sycophancy to a small neuron set + surgical correction). UNVERIFIED tension — full-text read owed before ever citing Fermion's negative as a law. |
| Formation law + access + 0.3% conversion | none found | The defensible novel core of their post. |
| Neuron-restricted weight tuning | **NeFT** arXiv:2403.11621 · TNF-DA (AAAI 2025) · **FiNE** arXiv:2503.01090 · MicroEdit · **Knowledge Neurons** arXiv:2104.08696 · Chang's neuron-tuning blog (Nov 1, 18 days prior) | Method class **KILLED** — 4-year-old pattern. |
| 2-neuron agent edit | MIT CSAIL weight-space programming (= our Research **648** Exp 8, Nov 2) | Same model, same benchmark; Fermion deltas: 64× smaller budget, 1 run vs 50, **20-decoy battery** (CSAIL lacks that rigor). A budget-efficiency point, not a new class. |
| ASR hallucination on non-speech | Barański arXiv:2501.11378 (phenomenon + Silero comparison); **Aparin arXiv:2606.07473** (hidden-representation detection/mitigation — closest); low-rank decoder projection (training-free); Calm-Whisper arXiv:2505.12969; SHALLOW benchmark | Phenomenon + mitigation framing heavily occupied; novel residue = the causal encoder-state *planting* methodology + single-neuron LID-beats-LID-model. |
| In-model VAD gate | US Patent 12198680B2; joint ASR+VAD training papers; domain-shift-gate lesson is classic | Bouncer = implementation flavor (10 KB reader off early conformer blocks) + the clean-vs-deployment gate lesson re-demonstrated. |

Surveys naming the landscape: Locate-Steer-Improve (2025), ACM CSUR MI survey, ACM CSUR knowledge-editing survey (ROME/MEMIT taxonomy), IJCAI 2026 actionable-interp survey.

## 3. Signal-diff vs what ships (§3.6 — every dismissal read at source)

| Shipped cousin | What it consumes | The census/edit instrument | Verdict |
|---|---|---|---|
| `mech_attribution` (katgpt-pruners, `cna_steering`; Research 053 ADOPT) | per-neuron activation differences between contrastive prompt sets (forward passes, no intervention) | interventional pushes + dose ladder + matched-random band; asks the causal question CNA only infers | **Complementary** — CNA is the cheap screen, the push census is the confirmation battery; both feed one artifact |
| `lens` lane (riir-infer, opt-in; Research 666 **DEMOTED for serving**) | weight-only hand-rule reads of frozen weights | census *runs forwards and intervenes*; its artifact is offline (no serving-cost claim — exactly the axis 666 died on) | **Different instrument, different posture** — 666's demotion binds serving claims, not offline census artifacts |
| `weight_edit_overlay` (katgpt-core 980, GOAT) + `closed_form_weight_write` (990) | authored spec (DirectionSubtraction/BypassRows) + halter-armed gate battery (norm-matched random control arm) | the **decoy battery is the two-sided upgrade**: site-random AND direction-random arms; the fader (`EditForm::ColumnScale`, down_proj output column at dose d) is a new EditForm beside DirectionSubtraction | **Composable** — battery row + edit-form row filed in 1241 |
| AHA (676, `matched_pair_authoring`, Issue 1224 LANDED) | self-supervised matched-pair directions (same-prefix resampled branches) | census supplies *site* priors (which layer, which neurons, dose) and a ranked candidate pool; AHA supplies *directions* | **The authoring-prior consumer** — the fusion 1241 exists to test |
| `sink_audit` (981) / `weight_audit` (1209) | KV captures / weight-structure features; offline census artifacts with fixture+tripwire | the lane shape to copy (capture → canonical artifact → digest → fixtures → gates), applied to push responses | **Shape precedent, different quantity** |
| `deletion_mask` (1208, `f32_apply.rs` `GateMaskOverlay`, BLAKE3-bound, apply-at-load) | sparse deleted-weight masks | the overlay artifact class for *edited-column* deltas (base + spec, never a fork) — the shipping rail for any neuron edit we author | **Artifact-family reuse** |
| reflex abstention stack (CorpusDistanceGate) | corpus-distance signals on modelless decisions | the L11 abstain dial is an *in-model* instrument — only meaningful on hosted/comparison lanes where a real LLM runs | **No wiring** — recorded; reflex engine is deterministic code |
| ESC gate / drift_guard / cascade worthiness | fitted sigmoid gates over serving signals | the bouncer lesson: gates trained on clean/benchmark distributions misfire on deployment tails (Bench 089's paraphrase-out-of-window is our own instance) | **Doctrine corroboration only** — the retrain recipe is a Plan 505 row |

## 4. Fusion + two-sided check (fusion priority ladder)

**Priority #1 (self-evolve product stack):** the census artifact's consumer is the **WSP/AHA authoring lanes** (overlay authoring = construction-time self-evolve). Honest sizing: the rulewright's text-rule corpus has *no* neuron-census consumer (its retrieval index is not a trained stack) — no refine-side wiring is filed; the decision family gets the Bench-089 corroboration only (stationary-surface law respected: no dynamics signal proposed anywhere).

**Priority #3 (inference-perf league):** Fermion is already a recorded league-watch row (Research 005 §5 F2, owner-gated). Neutrino/ternary-QAT context unchanged by this work; their neuron framework being open-sourced "soon" is the watch trigger for a future distill of the tools themselves (§0.5 clone at a pinned sha, never verdict from announcements).

**The fusion (filed as riir-infer Issue 1241):** census cell on the on-disk `Qwen3.5-0.8B-Base-Q8_0.gguf` (layer-prior-ordered: early + late bands first) → artifact with (site, direction-sign, dose, random-band margin) rows → (a) decoy battery beside `closed_form_weight_write`; (b) authoring-prior hand-off to AHA/matched-pair + probe authoring — GOAT = authoring cost/quality delta vs the no-prior baseline; (c) access check (base vs chat switch-set compare) as the free first cell. Kill bar pre-registered in the issue: planted-control recovery at the pre-registered margin or the issue closes NEGATIVE (ungoat discipline — no runnable proof path, no lane).

## 5. Path 0 + per-track verdicts

| Component | Coverage | Extraction |
|---|---|---|
| Interventional census + control battery + typed verdicts | none (CNA/lens/audits are different instruments) | **Extracted** → 1241 T1–T2 |
| Decoy battery | none (980 has one control arm) | **Extracted** → 1241 T4 |
| Layer/dose priors, density, formation detector, access check | none | **Extracted** (priors advisory-only) → 1241 T3/T5 |
| Fader edit form (`ColumnScale`) | none (DirectionSubtraction/BypassRows only) | **Extracted** → 1241 T4 rider |
| Running-light ablation (co-firing ≠ load-bearing) | corroboration (`influence_ring` design law; engagement-proof-first doctrine) | recorded, no lane |
| Probe-population overfit law (block-reader 27% vs single neuron ≥0.95) | corroboration of the probe-author discipline | recorded |
| Neuron-restricted tuning, finish-thinking, dose law, gate-retrain, ablation-census capacity, chat-unlock | riir-train pipelines exist; rows accumulate ≥3 | **Path 0.5** → Plan 505 (6 rows + 1 zero-cost prior) |
| Tempo/presence/langid readouts, bouncer, encoder-state planting | audio lane owner-DEFERRED (riir-infer Issue 015) | **Pass** — redirect to Research 005; revisit only when Issue 015 unblocks |

**Novelty gate:** Q1 no (class published; internal fusion value is instrument-class) · Q2 no (no new capability class — the WSP lanes already author edits) · Q3 partial (the authoring-prior selling point is real but unproven until 1241 runs) · Q4 yes (connects census × WSP × deletion-mask artifact family × steering family) → **not Super-GOAT; GOAT gated behind 1241's pre-registered bars; top verdict today = Gain.**

**MOAT gate:** katgpt-rs row (base primitive): the *scoring* half (AUROC-vs-scrambles screen, typed-verdict battery, decoy battery) is modelless and dep-free — it belongs in katgpt-pruners `mech_attribution`/katgpt-core beside the overlay lanes when 1241's poc proves the fusion; the *runner* (forwards over checkpoints, artifact lanes) is riir-infer substrate, its home per the MOAT table (weights/quant/encoders/census). No private-tier content anywhere in this filing (public research + public substrate).

## 6. Verdict

**Gain** — instruments + intelligence + a training backlog; every lane proposal carries its proof path or files no lane (ungoat discipline).

- **Track verdicts (separate, never pooled):** offline census instrument → Gain, `riir-infer/.issues/1241` (poc + kill bar) · edit-form/battery riders → rows in the same issue · training recipes → Path 0.5, `riir-train/.plans/505` (backlog; unqueued; heavy cells route off m3) · speech/readouts → **Pass** (PASS-Redirects below) · persona/atlas intelligence → recorded here, no lane.
- **Files created this session:** this note · `riir-infer/.issues/1241_neuron_census_authoring_prior.md` · `riir-train/.plans/505_fermion_neuron_edit_recipe_backlog.md` · PASS-Redirects into 666, 053, 648 (this repo) and `riir-infer/.research/005` (speech).
- **Standing watch:** the Fermion framework open-source ("soon") is the trigger to clone-and-mine the tools at a pinned sha; `FermionResearch/Qwen3.5-0.8B-edited` is the 2-neuron-edit reference artifact if 1241/505 ever need a known-good target to chase.

## PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Fermion Research [no arXiv — "Mapping every neuron in Qwen3-8B" + "Reading the neurons of a speech model", fermionresearch.com, 2026-11-16/18] — interventional census instrument + decoy battery + training backlog distilled to 684; speech readouts pass to riir-infer Research 005 (audio lane owner-deferred, Issue 015).

## Provenance

- Pages fetched over the wire 2026-11-20; all numbers are **their measurements on their harnesses** — none reproduced here. The `Qwen3.5-0.8B-edited` card is the only artifact-level claim (2,048 weights, layer 19, columns 353/803; Apache-2.0).
- Two prior-art agents (28 + 25 queries, verdicts + IDs above) and both adversarial advocates ran in one parallel batch 2026-11-20; advocate-reported facts cited here were coordinator-checked against repo AGENTS.md records (Issue 1208 artifact names, 981 Bonsai pin, Issue 691 KL fix, Bench 089 rate-guard instance) — GPU-hour figures in Plan 505 are advocate estimates, labeled as such.
- Internal-first sweep: `Fermion|neuron atlas|neuron tuning|steering vector|single neuron|SAE|monosemantic` over riir-infer `.research/.plans/.issues` → hits only Research 001/005 (the prior Phonon distills); `Phonon-2.5|Detta|bouncer|phantom` workspace-wide → only 005. WSP-family census via vocab.md §10 (the 648 keyword-miss lesson applied).
- Live-state citations verified against the repos' current AGENTS.md records this session (1208 landed-code posture, 981 UNDECIDED pin, 1224 landed, 666 demotion, Issue 015 defer).

## Landed — katgpt-rs Issue 1237 (2026-11-20, `65914d256`)

The modelless scoring halves shipped opt-in (no new feature flags): `katgpt_pruners::mech_attribution::census_screen` (`screen_auroc` — 20 scrambled-label population searches, admit at null max + 0.20, zero-alloc; `control_verdict` typed SwitchFound / NoSwitch{band_measured} / Undecided{under_floor}, `VerdictTally` never pools) and `katgpt_core::weight_edit_overlay` (`EditForm::ColumnScale` fader, lerp-exact endpoints, stale-base refusal; `decoy_battery::run_decoy_battery` two-sided on one frozen read). All G1 canaries + G4 counting-allocator gate green at fixture scale. **Not a value claim**: the kill bar stays with riir-infer Issue 1241 T2 — if that census cell closes NEGATIVE, these demote with it. Issue file removed (git history).
