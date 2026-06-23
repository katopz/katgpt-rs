# Claim Rubric Audit — Research Notes vs. `Claim` Fixtures (Plan 307 T4.2)

**Date:** 2026-06-23
**Plan:** [katgpt-rs/.plans/307_claim_rubric_runtime.md](../.plans/307_claim_rubric_runtime.md) (Phase 4 task T4.2)
**Research source:** [katgpt-rs/.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md)
**Runtime:** `katgpt_rs::claim_rubric` (feature `claim_rubric`, `src/claim_rubric/`)

---

## TL;DR

T4.2 asks: "every probe/steering research note that invokes L1/L2/L3 vocabulary must now link to a `Claim` fixture in its corresponding primitive's test file." This audit grep'd `katgpt-rs/.research/*.md` for the rubric vocabulary (`\bL1\b`, `\bL2\b`, `\bL3\b`, `causally controls`, `mechanistically mediates`, `induces`, `reliably produces`, `functionally steers`, `behavioral evidence`, `functional evidence`, `causal-mechanistic`, `predict-control parity`, `predict-control discrepancy`).

**Honest result: the rubric vocabulary is currently confined to R287 itself.** The seven cousin notes that R287 §4 scores (CNA, EmotionDirections, PosteriorGuided, FaithfulnessProbe, magnitude-drift, FPCG, spectral) do NOT yet invoke L1/L2/L3 in their own bodies — they pre-date the rubric. So the "backfill gap" is not "notes use the words and lack fixtures"; it is "notes do not yet use the words at all, and the corresponding primitives' fixture coverage is concentrated in one file (`tests/claim_rubric_test.rs` + `tests/bench_307_claim_rubric_goat.rs`), not per-primitive."

This is a gap audit, not a victory lap: the rubric is green for the seven §4 rows, but its *adoption* by the cousin notes is zero. The recommended follow-ups are therefore about adoption, not fixture count.

---

## Audit Method

1. Grep `katgpt-rs/.research/*.md` for the vocabulary patterns above (`include_pattern = "katgpt-rs/.research/*.md"`).
2. Distinguish **signal** (probe/steering evidence-ladder usage) from **noise** (cache tiers like "fits in L1/L2 cache", `‖·‖₂` L2 norms, `L1` MLP layers, generic verb "induces a distribution"). Noise matches were discarded.
3. For each signal match, identify the primitive and check whether a `Claim::` fixture exists for it in `katgpt-rs/tests/` (via `grep` for `Claim::new` / `claim_rubric::Claim`).
4. Cross-reference against the R287 §4 table (the seven primitives that DO have fixtures per Plan 307 Phase 2).

---

## Fixture Coverage Table

### Group A — R287 §4 primitives (Phase 2 fixtures exist)

These seven primitives have `Claim::new(...)` fixtures in `katgpt-rs/tests/claim_rubric_test.rs` (round-trip tests) and `katgpt-rs/tests/bench_307_claim_rubric_goat.rs` (GOAT gate). They are the single source of truth for the §4 scores.

| Research note | Primitive | Vocabulary used | `Claim` fixture exists? | Gap / action |
|---|---|---|---|---|
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 1 | `EmotionDirections::project` (Plan 162) | L1, L2, L3, predict-control parity | ✅ `claim_rubric_test.rs::r287_s4_emotion_directions_project_is_l1` + GOAT fixture | None — fixture is the SoT for the L1 score. |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 2 | CNA contrastive (Plan 087) | L1+, L2, L3, predict-control parity | ✅ `claim_rubric_test.rs::r287_s4_cna_contrastive_is_l1` + GOAT fixture | None — fixture models L1+ → honest L1. |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 3 | `FaithfulnessProbe::behavior_delta` (Plan 278) | L2 candidate, specificity control, competing-explanation | ✅ `claim_rubric_test.rs::r287_s4_faithfulness_probe_behavior_delta_is_l1` + GOAT fixture | None — fixture models L2 candidate → honest L1 (specificity TBD). |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 4 | `FutureBehaviorProbe` / FPCG (Plan 292) | L1 (planned), L2 Pareto, predict-control parity | ✅ `claim_rubric_test.rs::r287_s4_future_behavior_probe_is_l1` + GOAT fixture | None — fixture models L1 (planned, blocked on Issue 032). |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 5 | `PosteriorGuidedPruner` (Plan 239) | L1–L2, generalization across regime shifts | ✅ `claim_rubric_test.rs::r287_s4_posterior_guided_pruner_is_l2` + GOAT fixture | None — fixture models upper bound L2. |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 6 | HLA `evolve_hla` | L1, magnitude-drift L1 finding | ✅ `claim_rubric_test.rs::r287_s4_hla_evolve_is_l1` + GOAT fixture | None — fixture models L1 (no downstream-causal claim). |
| [R287](../.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md) §4 row 7 | Spectral probes (EGA / SpectralQuant / irrep) | L1, L2 downstream, predict-control parity | ✅ `claim_rubric_test.rs::r287_s4_spectral_probes_is_l1` + GOAT fixture | None — fixture models L1 eigenbasis read. |

**Group A status: 7 / 7 fixtures present, all backed by the GOAT gate. Phase 2 green.**

### Group B — Cousin notes that R287 references but do NOT yet self-invoke the vocabulary

These are the notes R287 §8 explicitly cites as "primitives scored in §4". They pre-date R287 and therefore do not use L1/L2/L3 vocabulary in their own bodies. A `Claim` fixture exists for each *as scored in R287*, but there is no per-primitive test file owning that fixture — it lives in the shared `claim_rubric_test.rs`. This is the actual adoption gap T4.2 surfaces.

| Research note | Primitive | Vocabulary used in-note? | `Claim` fixture exists? | Gap / action |
|---|---|---|---|---|
| [R053 CNA](../.research/053_CNA_Contrastive_Neuron_Attribution.md) | CNA contrastive | ❌ No L1/L2/L3 usage | ⚠️ Indirect (shared fixture in `claim_rubric_test.rs`) | Note does not yet state its evidence level in-header. |
| [R144 EmotionDirections](../.research/144_Functional_Emotions_Linear_Representations_Behavior_Control.md) | `EmotionDirections::project` | ❌ No L1/L2/L3 usage | ⚠️ Indirect (shared fixture) | Same — no header-level evidence declaration. |
| [R211 PosteriorGuided](../.research/211_Bayesian_Agent_Posterior_Guided_Skill_Evolution.md) | `PosteriorGuidedPruner` | ❌ No L1/L2/L3 usage | ⚠️ Indirect (shared fixture) | Same. |
| [R244 FaithfulnessProbe](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md) | `FaithfulnessProbe::behavior_delta` | ❌ No L1/L2/L3 usage | ⚠️ Indirect (shared fixture) | Same — note calls itself "causal intervention" without L3 evidence; rubric would flag this. |
| [R267 FPCG detect-vs-predict](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md) | `FutureBehaviorProbe` | ❌ No L1/L2/L3 usage (R287 fuses *with* it; R267 itself uses "detection vs prediction" framing, not the ladder) | ⚠️ Indirect (shared fixture) | R267 is the *vocabulary source* R287 grades, but does not self-grade. |
| [R286 magnitude drift](../.research/286_Attention_Drift_Depth_Invariance_Diagnostic.md) | HLA freshness confounder | ❌ No L1/L2/L3 usage (R287 §2.1 cites it as "an L1 finding *about* a latent kernel") | ⚠️ Indirect (folded into the HLA fixture) | R286 itself is a behavioral finding; it has no fixture because it is a *confounder*, not a primitive. Out of scope for a fixture. |

**Group B status: 0 / 6 cousin notes self-invoke the rubric vocabulary. All six have *indirect* fixture coverage via R287's shared test file, but none owns a `Claim` fixture in its primitive's own test file.**

### Group C — Incidental vocabulary hits (noise, no action)

For completeness — these `.research` files matched the grep but are not probe/steering claims and do not need fixtures.

| Research note | Match | Why noise |
|---|---|---|
| R008, R020, R022, R024, R029, R031, R037, R044, R065, R066, R067, R070, R077, R079, R086, R092, R094, R110, R115, R124, R125, R137, R139, R142, R150, R151, R166, R171, R172, R176, R185, R191, R192, R195, R196, R205, R230, R231, R233, R234, R236, R238, R243, R246, R248, R249, R255, R257, R265, R266, R268, R270, R274, R275, R276, R284 | "L1/L2/L3 cache", `‖·‖₂` norms, MLP layer indices, "induces a distribution/prior" (non-rubric sense), smooth-L1 loss, Rényi entropy tiers, etc. | All are CS/ML shorthand unrelated to the evidence ladder. The grep regex is intentionally broad to catch overclaims; these are the false positives. |
| [R016 AutoTTS](../.research/016_AutoTTS_Dynamic_Test_Time_Scaling.md) L89 | "fine-grained behavioral evidence" | Generic phrase, not the rubric's L1 "behavioral evidence level". |

**Group C status: no action. Documented only so a future narrower grep can reuse this list.**

---

## Pre-existing naming clash (noted, not in scope)

`tests/bench_284_clr_goat.rs` uses a *different* `Claim<T>` (the `clr` typed claim, not `claim_rubric::Claim`). This is the same clash Plan 307 T1.2 documented when it decided NOT to re-export `Claim` at the crate root. No action here — flagged for awareness only.

---

## Recommended follow-ups

These are filed as future work; T4.2 explicitly does not create fixtures, it documents gaps.

1. **Header-level evidence declaration for the six cousin notes (R053, R144, R211, R244, R267, R286).** Each should add a one-line `> **Evidence level (per R287):** L1` (or L2 candidate, etc.) to its header block, mirroring R287's own header style. This is the *adoption* gap, not a fixture gap — the fixtures already exist centrally. Highest leverage because it makes the rubric's vocabulary propagate into the notes that currently don't use it.
2. **Per-primitive test ownership.** Today all seven `Claim` fixtures live in one shared file (`claim_rubric_test.rs`). When a primitive's own test directory grows (e.g. `tests/posterior_guided_rubric.rs`), the fixture should migrate there so the primitive owns its own claim score. Low urgency while the shared file is small.
3. **R244 FaithfulnessProbe self-grade reconciliation.** R244's title ("Causal Intervention on Injected Memory") uses L3-flavored language ("causal intervention") while R287 §4 honestly scores it L2 candidate → L1. A one-paragraph reconciliation in R244 — either downgrade the title vocabulary or add the missing specificity control — is the single highest-value rubric application in the corpus today.
4. **Re-run this audit after the next probe/steering note lands.** R292 (FPCG Phase 4 GOAT) is the first note R287 §7 mandates must run the §5 checklist. When it ships, this audit should be re-grepped to confirm R292 self-declares its level and links back to the FPCG fixture.
