# Research 653: Probe-Authored Repair × twinarm — the Specialist-Head Consumer Fusion (649's first veto-passing consumer, class-1 leg)

> **Source:** user-directed fusion (2026-11-05): Research 649 (probe-authored circuit
> grammar) × the twinarm specialist trainer pipeline (`riir-ai/.docs/03_pillars/twinarm_pipeline.md`,
> mechanics = public code in `riir-instinct/src/instinct_specialist.rs`) ×
> riir-rethink as consumer context. Follows Research 650's sweep — which covered the
> ten supergoat-candidate docs but NOT twinarm — and executes 650's recorded flip path:
> *"a weight-attribution study on ONE named failure family re-opens 649 §8 on its own terms."*
> **Date:** 2026-11-05 (rev 2 same day — verdict round 1 REVISE addressed: collision-repair
> leg retracted on the shared-column argument; fair-baseline controls replace the strawman;
> oracle census replaces zero-the-bucket; Mode A reworded; salvage targets stated plainly.
> rev 3: verdict round 2 REVISE addressed — pre-registered tune/eval split binds G1+G2
> (selection and scoring separated, the authoring-level one-read law); G0 oracle batched
> per-suite. rev 4: verdict round 3 REVISE addressed — the split law re-specified as
> PoC-local proportional alternation (the shipped round-robin laws take a budget only,
> no seed; and their balanced-eval/skewed-tune output is a distribution shift the
> prior-sensitive baseline must not face)
> **Status:** Active — fusion filed; PoC issue `riir-instinct/.issues/028_twinarm_probe_repair_poc.md`
> carries the proof path with pre-registered gates; tier FINALIZES with the PoC (fold law pre-registered)
> **Related Research:** 649 (the anchor primitive), 650 (the prior sweep + rethink blockers), 648 (parent)
> **Related Issues/Plans:** `riir-instinct/.issues/028` (the PoC); katgpt-rs Issue 980 /
> Bench 978 (the landed substrate); Bench 979 (the two-region smallest step)
> **Classification:** Public (mechanics cited at public-code level; private-record facts cited by pointer only)

---

## TL;DR

The twinarm lane is the consumer Research 649's §8 promotion gate was waiting for, and the
one surface Research 650's sweep did not cover. The specialist heads are **linear one-vs-all
logistic heads over the floor's own FNV-1a hashed n-gram bags** (`VOCAB = 1<<17`,
`bag_into`, i8 weights + per-class f32 scales) — the simplest possible circuit class, where
649's probe stats run **closed-form over the corpus** (no rollout infrastructure: the
"activations" are bucket counts and per-class logits, both computable directly), the
**outcome-split pools are load-bearing instead of degenerate** (classification rows have
argmax≠gold failing pools — the extension point 649's RL fixture could not exercise becomes
the main path), and **the behavior referee already ships** (the twinarm gates + the
frozen-read winner law + RISP seal + `winner_bridge`), sighted — unlike rethink's ESC lane,
whose blindness blocked 650's rethink rows. **The repair target is the documented
weak-teacher collapse class** (riir-train Bench 609, private record — a distilled arm lost
its suite by a wide margin and the winner law refused it): the distilled arm fits teacher
soft targets, so its weights encode the teacher's errors as systematic bucket→class arcs
that **gold-labelled outcome pools can see**, parse into a 649 fn-grammar record (BLAKE3
sidecar beside the RISP artifact — the RIDT sidecar pattern, no format break), and repair
from the discrete `EditForm` vocabulary over a probe-derived strength window — **with the
repair required to beat fair weight-side baselines** (continue-train-on-gold, per-class
bias recalibration, per-bucket coordinate search, all tuned on the same authoring holdout)
before "probe-authored" buys anything beyond attribution. A second class — hashed-bucket
collision interference — is **demoted to census + attribution record only** (rev 2): two
colliding tokens share one weight column, so no weight edit can decouple them, and convex
one-vs-all logistic training already fits the best compromise for that column — the honest
statement is *"collision cost is structurally unremovable at frozen width"*, and the
census measures its true cost with an **un-collided oracle** (a diagnostic-only retrain
with the flagged pair split into separate buckets, never shipped) rather than a
zero-the-bucket delta, which conflates importance with interference. **Verdict: GOAT-eligible
fusion on the class-1 leg (Q1 qualified, same tier as 649/650 filings); tier finalizes only
when the PoC gates pass — the oracle census folds class 2 pre-PoC if the ceiling measures
≈0, and the fixture gate folds the whole lane if the repair cannot beat its fair baselines**
(the issue-205 pattern, pre-registered).

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable claim is that 649's three-stage author (gate scan → wire slope →
behavior-verified emission over a derived strength window) generalizes from RL-policy MLPs
to **any trained linear head whose input space is hashed and whose error is target-noise
shaped** — because the tap source collapses to a corpus scan and the outcome pools supply
the gate variable for free. The honest boundary (rev 2): weight edits move shared columns
wholesale, so the class the repair can address is *encoded wrong arcs* (target-noise
degeneracies like teacher poison), never *input-space aliasing* (collisions). The
katgpt-core substrate stays consumer-free by default; the linear-head pool-contrast variant
rides the consumer's pull (an opt-in `probe_repair` feature in `riir-instinct` forwarding
`katgpt-core/probe_circuit_grammar`), per the 980/649 discipline.

---

## 1. The fusion, sharpened

| 649 lane (RL fixture: 8.7K-param policy MLP) | twinarm head (linear over hashed bags) | adaptation |
|---|---|---|
| `StepTap { gate_var, src, dst_pre }` rollout taps | **closed-form**: src = the L2-normalized sparse bag; dst_pre = per-class logits W·x — the head IS one layer; taps are a corpus scan, not a rollout | tap infrastructure unnecessary |
| `ProbePools` failed/succeeded outcome split — **degenerate** in the fixture (all-failed pools) | failing pool = rows with argmax ≠ gold; succeeding pool = argmax = gold — **load-bearing** (gold labels exist for every authoring row, so teacher-poisoned weights are *visible* to the pools) | 649's extension point becomes the main path |
| gate scan: one-sided magnitude edge-contrast over gate bands | **outcome contrast**: bucket contribution contrast between failing and succeeding pools (band = {fail} vs {succeed} — the binary degenerate band) | same statistic, degenerate band |
| gap row: `argmax` readout-row weight diff | the wrong class row is **known** from the failing argmax; ties fall back to the 649 readout-argmax | simpler |
| wire slope: pooled LS of `dst_pre[gap]` on `src[drive]` | pooled LS of the wrong-logit contribution on bucket count across rows → **centers the strength window** (649's slope-confound caution transfers verbatim: the sweep decides, the slope only centers) | same stat |
| `derive_strength_grid` + `EditForm::DirectionSubtraction` + `emit_word` dose sweep, failure-rate acceptance | same forms on `W[c*][j]`; acceptance = authoring-holdout repair bar **that must also beat the fair weight-side baseline battery** (§3) | referee is stricter than 649's |
| 980 freeze-envelope provenance trailer | RISP artifact + **BLAKE3 grammar sidecar** (the RIDT `.blake3` sidecar pattern; no format break) + monotonic version bump on re-seal | sidecar, A8-clean |
| behavior referee: authoring episode failure rate | **already ships**: lr=0 control, distractor law, authoring holdout, and the frozen-read winner law (paired LB95 > 0 vs A0) | the referee is the consumer's own |

Two repair modes, both offline (A8: never a runtime edit — 650's OFT row confirms the
ladder is over artifact versions):

- **Mode T (trainer-state):** repair the f32 weights pre-seal; re-seal; full gate battery.
- **Mode A (artifact-direct):** ±integer-strength edits on the i8 rows of an existing
  sealed artifact. **Requires a labelled authoring slice** (the arena's holdout rows —
  the referee needs labelled pools wherever it runs); what it skips is the **full training
  corpus**, not labels. Re-seal + monotonic version bump. Secondary posture.

## 2. Why the twinarm lane passes the veto gates 650 confirmed

Research 650's cross-cutting findings blocked every rethink-aimed fusion on three facts:
ESC is N=1 armed (`emotion_syn` only); rethink's documented degeneracies are
**gate-layer threshold/mask shapes** (Bench 062/079/0057), repaired gate-side, none
head-shaped; and `esc_voi`'s value legs are receipts-only blind until the decstat outcome
join. **None of the three binds here:**

1. **The board is 8 suites, not N=1.** The twinarm heads seat in instinct's serving board
   (public AGENTS.md): specialists seat via H2/NBSVM rows (banking77 nbsvm v2, typed,
   ag_news, massive); **emotion, xnli_en, sst5 serve A0/A1 — no specialist seat**.
2. **A documented weight-shaped degeneracy exists** (649 §8's precondition): the trainer
   lane's recorded weak-teacher collapse (riir-train Bench 609 — private record; class
   fact: a distilled arm collapsed on one suite and the winner law refused it; teacher
   train acc predicts transfer direction). The weights are the pathology; the referee
   (winner law) is sighted on labels — no blindness blocker.
3. **Taps are closed-form.** 649 §8's second precondition (tappable forwards) is met
   trivially: a linear head's activations are arithmetic over the bag, and the tokenizer
   law is already transcribed trainer-side (`bag_into`, `VOCAB = 1<<17`).

## 3. The degeneracy classes (rev 2: one repair target, one census)

**Class 1 — weak-teacher collapse: THE repair target.** The distilled arm minimizes BCE
against `t = mix·onehot(gold) + (1−mix)·q_teacher`; a weak teacher's soft targets write
its errors into the weights as systematic bucket→class arcs. These arcs are visible to
gold-labelled pools (the authoring rows carry gold), localized by the outcome contrast,
and addressable by a discrete direction-subtraction word on `W[c*][j]`. **The honest
boundary (verdict round 1):** the one-vs-all logistic objective is convex, so the trained
weights are already the best compromise for their targets — the repair can only gain
where training stopped short of the gold optimum (the target noise IS the gap), and
generic weight-side alternatives exist. The repair therefore must beat a **fair baseline
battery, all tuned on the same authoring holdout**, before "probe-authored" buys anything
beyond attribution: (b1) continue-train on gold for k steps (the generic move toward the
gold optimum), (b2) per-class bias recalibration, (b3) per-bucket coordinate search.
Paired LB95 > 0 vs the best baseline = the value gate; the attribution record lands
regardless.

**Class 2 — hashed-bucket collision interference: census + attribution record ONLY
(retracted as a repair target).** Two tokens sharing a bucket share the weight column
`W[·][j]` for every class; **any edit to that column moves both tokens' contributions
identically** — a weight edit cannot decouple them, and with the lexicon frozen
(lexicon-fork guard) no feature-side split is available either. Convex training already
fits the best compromise for the shared column, so a "collision repair" would just move
the compromise toward whichever token dominates the edit's optimizing pool — generic
post-hoc tuning wearing a collision name. **What remains is honest measurement:** the
census estimates the true collision cost with an **un-collided oracle** — a
diagnostic-only retrain with flagged colliding pairs split into separate buckets (a
different feature map, never shipped) — giving the ceiling any future remedy (a width
bump, a lexicon fork) would have to beat. Oracle ≈ 0 on every suite ⇒ class 2 folds as a
recorded negative ("collision cost immaterial at 2¹⁷ on these corpora"); oracle material
⇒ the census record documents a structurally unremovable cost of the frozen feature law —
input for the lexicon-fork-guard's own cost accounting, never a weight-edit claim.

## 4. Prior art

Searches this session (2026-11-05): hashed-feature collision repair / post-training
linear-classifier weight editing / feature-hashing collision mitigation. The composition is
unpublished; the components are, per 649 §2 + 650 §6 (AI-Lancet, ROME, INNER, ASPD,
RASP/Tracr/ALTA, Provable Polytope Repair/REASSURE, cleanlab, CAP, DRAGO) plus the new
register rows:

- **Feature-hashing collision effects** — Weinberger et al. 2009 lineage; Booking.com 2018
  blog; sklearn #7513 ("collisions are equivalent to creating new polysemy"); hash
  embeddings (arXiv:1709.03933). Phenomenon + train-time remedies only (bigger space,
  signed hashing, learned collision resolution); **none measures a frozen-width cost
  ceiling via an un-collided oracle** — the census instrument's delta — and none repairs
  weights (which §3 shows would be void anyway).
- **INNER / AI-Lancet** — nearest repair neighbors; gradient/masked-GD, CNN classifiers,
  no discrete vocabulary, no trainer integration, no attribution record.
- **ROME/MEMIT-line model editing** — transformer fact edits, solved per-edit from data;
  never a discrete-vocab emission inside a specialist trainer's gate pipeline.
- **Post-hoc linear-head tuning** (bias recalibration, temperature/bias correction
  literature) — the fair-baseline battery's members; standard practice, not competitors
  to the fusion's claim (the claim is repair-attribution + baseline-beating emission, not
  "post-hoc tuning is new").

**Pinned claim (pre-search):** post-training, probe-authored repair of the weak-teacher
collapse class in trained linear specialist heads (one-vs-all logistic over FNV-1a hashed
bags) inside a twin-target trainer — gold-pool contrasts parse the teacher-poisoned
bucket→class arc into a grammar record sealed beside the artifact, a discrete edit
vocabulary emits the fix over a probe-derived strength window, authoring-holdout-verified
**against fair weight-side baselines**, finally gated by the existing frozen-read winner
law; plus a collision census whose un-collided oracle prices the frozen feature law's
aliasing cost.

## 5. The proof path (pre-registered; lives in `riir-instinct/.issues/028`)

Gate names are unified with the issue: **G0 census · G1 fixture · G2 value · G3
no-regression · G4 attribution.**

- **G0 oracle census (base-rate gate, runs FIRST — the issue-205 pattern):** per suite,
  bucket member tables over the trainer's own tokenizer law; flag colliding pairs with
  two high-support class-disagreeing members; collision cost measured by the
  **un-collided oracle** — ONE diagnostic retrain per suite that splits ALL flagged
  pairs into separate buckets at once (a different feature map, never shipped), giving
  the suite's total collision ceiling; per-pair retrains only if that total is material.
  **Oracle ≈ 0 on every suite ⇒ class 2 folds as a recorded negative** (class 1
  unaffected). A level readout — no dynamics signal, stationary-test-clean.
- **The tune/eval split (pre-registered, binds G1 AND G2):** the authoring holdout is
  split ONCE, before anything trains, by a PoC-local deterministic rule —
  **proportional alternation**: within each label's authoring rows in first-appearance
  order, odd positions (1st, 3rd, …) go to the tune slice and even positions (2nd, 4th,
  …) to the eval slice. No RNG and no seed constant — none exists to pin: the reflex
  `stratified_split` and the trainer's `stratified_holdout`
  (`instinct_specialist.rs:425`) are budget-only laws whose docs say "no seed to
  record"; the rule itself is the pin. Per-label proportions hold in BOTH slices (each
  label contributes ⌊n_label/2⌋ vs ⌈n_label/2⌉ rows), asserted by a unit test before
  any arm runs. Deliberately NOT the shipped `stratified_holdout` round-robin: that
  law fills its budget one-row-per-label-per-round, yielding a label-BALANCED eval
  slice and the SKEWED remainder as tune — a selection/scoring distribution shift that
  a prior-sensitive arm (b2, bias recalibration) is most exposed to; the shipped law
  stays untouched for seat-time holdouts. Every arm — the repair's dose sweep and
  b1–b3 alike — selects parameters on the tune slice ONLY; G2's paired read happens
  once on the eval slice. Without this, the value gate reads on the selection set —
  the one-read problem the winner law exists to prevent, at authoring level.
- **G1 fixture (teacher-noise collapse, the class-1 leg):** synthetic corpus with a
  pinned corrupted-teacher construction (soft targets flipped toward specific
  bucket-correlated wrong classes; constants pinned before the run); train Arm A (gold
  reference) and Arm B (corrupted teacher); verify the gold-pool parse recovers the
  poisoned arcs (bucket + wrong class row) exactly; dose-sweep repair selected on the
  TUNE slice. Gates: zero-strength bit-identity (the lr=0-control analog); isolation
  (untouched weights byte-identical, unaffected-row predictions unchanged — the 649
  battery analog); and the **fair-baseline battery** (b1 continue-train-on-gold, b2
  bias recalibration, b3 per-bucket coordinate search), each tuned on the same TUNE
  slice only, which the repair must beat (paired LCB95 > 0 read on the EVAL slice) for
  the "probe-authored" claim — a baseline-beating repair is the pass, a merely-adequate
  repair still lands the attribution record but fails the lane's value claim.
- **G2 value (real suites):** repair where class 1 applies (Arm B heads with a surviving
  teacher dump, or a re-captured one); repaired arm strictly beats its unrepaired base
  AND the best baseline — paired LB95 > 0, read ONCE on the eval slice (never the tune
  slice any arm selected on); **seating claims
  additionally require the existing one-frozen-read winner law** (no second read, no
  threshold iteration against test — plan-016 §6 discipline adopted). The honest seat
  target is **sst5** (the one suite where a specialist plausibly crosses the served A1
  bar); emotion (A0 0.8850 — every challenger measured out) and xnli_en (bag-view
  representation ceiling) are attribution-record targets, not seat targets.
- **G3 no-regression:** non-targeted artifacts byte-identical.
- **G4 attribution:** grammar sidecar round-trips, digest-verified, stale-refusing, names
  the G1 arcs correctly.
- **Fold law:** G0 class-2-oracle ≈ 0 ⇒ class 2 folds (census-only negative); G1
  failure ⇒ the whole lane folds into the trainer, no zombie module, note demoted in the
  same commit (the 649 demotion law verbatim).

## 6. Consumer check (the priority-1 two-sided question)

- **Decision-serving family:** attribution-carrying repair of near-miss/distilled arms
  (every repaired artifact carries a BLAKE3 grammar record naming the arcs it fixed — a
  new provenance class for the serving family); a salvage seat only where the winner law
  says so (sst5 is the honest candidate; emotion/xnli are expected to stay
  floor-served — the record explains why in machine-readable form). **Rethink context
  (honest framing):** rethink consumes this stack downstream — seated specialists
  strengthen the cheap legs the ESC lane composes against the encoder think leg, and the
  grammar record is a future decstat outcome-join surface (report-only; **no ESC arming,
  threshold, or rethink code change is claimed or needed by this filing** — 650's rethink
  blockers stand untouched; this fusion simply does not route through them).
- **Rulewright:** the rerank seat (refine Plan 225 / Bench 119) reranks fixer candidates
  through instinct's `Ext` seat — a repaired specialist head is a better rerank head
  candidate; the heal head shares the hashed-bag shape, so the same census applies there.
  Follow-up, not claimed.

## 7. Honest caveats

1. **The class-1 gain ceiling is bounded by the target-noise gap** — convexity means the
   repair competes with generic moves toward the gold optimum; if continue-train-on-gold
   matches it, the lane's value is attribution-only and the tier records exactly that.
2. **The teacher dump's on-disk survival is unverified** — T2's Arm-B leg is conditional
   on a Bench-609-era RIDT dump existing or being cheaply re-capturable; the fixture leg
   (G1) is unconditional.
3. **Collision cost may measure ≈0** (Zipf tails) — the oracle census folds class 2 as a
   recorded negative; and even a material ceiling is NOT a repair opportunity (§3), only
   a priced property of the frozen feature law.
4. **The slope-confound caution transfers verbatim** — the LS slope centers the window;
   only the dose sweep with the authoring referee picks the word.
5. **Mode A lives at i8 granularity** — the window derives from integer strengths; finer
   control needs Mode T; and Mode A needs a labelled authoring slice wherever it runs.
6. **Single-word scope** — the 649 single-arc limitation transfers; multi-bucket repairs
   are sequential words with recorded order-dependence (Bench 979's far-region-first
   finding is the precedent for ordering discipline).
7. **Salvage seats are thin** — sst5 is the one plausible seat; emotion/xnli are
   attribution-only (§5 G2). The lane's headline value is the attribution class +
   baseline-beating repair, not a wave of new seats.
8. **katgpt-core stays consumer-free by default.** The pool-contrast generalization of
   the 649 stats files as a katgpt-rs issue ONLY if G1 passes (the parse needs it; the
   census G0 does not) — the consumer's pull, never a speculative substrate edit.
9. **Private-number hygiene** — collapse-class facts cited by pointer to private records;
   quoted numbers only from public board rows.

## 8. Verdict

**GOAT-eligible fusion on the class-1 leg (Q1 qualified — components published, the
composition unpublished; Q2/Q3/Q4 solid: repair-and-seat-with-attribution is a behavior
class no stack lane has, the selling point finishes ("seated specialists carry a sealed
circuit-grammar record; teacher-poisoned arms get a deterministic, baseline-beaten repair
pass that must still win the frozen read"), and the fusion consumes 649's substrate + the
twinarm trainer + the winner law + RISP seal ≥ 2 pillars).** Tier FINALIZES with the PoC:
gates pass ⇒ GOAT (fusion) with the bench record; census-oracle ≈ 0 folds class 2 only;
G1 failure folds the lane ⇒ recorded negative, note demoted in the same commit (ungoat
discipline). **MOAT gate:** code home = `riir-instinct` (the trainer lives there; public
crate; opt-in feature forward), substrate = `katgpt-core` (consumed, never forked — zero
probe-grammar vocab in instinct src verified this session), this note = the public fusion
record; rethink = consumer context only, no code edge claimed. **Revision history (verdict
round 1):** the collision-repair claim was retracted on the shared-column argument (two
colliding tokens share `W[·][j]`; no weight edit decouples them; convex training already
holds the best compromise); the strawman retrain control was replaced by the fair
weight-side baseline battery; the zero-the-bucket census metric was replaced by the
un-collided oracle; Mode A's "without corpus" was corrected to "without the full training
corpus (labels still required)"; salvage targets were stated plainly (sst5 or nothing).
