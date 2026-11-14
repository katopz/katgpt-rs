# Research 671: dspy-wsp — Decision-Model Weight-Space Programming (the Head Surface × Direct-Read Closed-Form Cell)

> **Source:** cmpnd.ai blog "Weight-Space Programming with DSPy" (2026-11-13) + `github.com/cmpnd-ai/dspy-wsp` (MIT) — both fetched in full 2026-11-14, NO clone (§0.5: a product README + blog, quote-pinned to the fetched text; every number below is theirs). Productizes Takehi/Teoh/Zhang/Khattab "Sparks of Weight-Space Programming" (MIT CSAIL 2026-11-02 — [Research 648](648_Weight_Space_Programming.md)) on DECISION models, with Haoze "Hector" He's earlier decision-model editing (`llm-honing`) as the named class prior.
> **Date:** 2026-11-14
> **Status:** Active — **Gain** (external measured-evidence + one untried-cell finding; no novelty claim — the class is published)
> **Related Research:** 648 (WSP parent — the obstacle-vs-ability law), 664 + 659 + 668 (the author-solver siblings: LP / MEMIT ridge / discrete-Ising), 649 (probe author — the no-LLM posture), 562 + 573 (the TypeSafe/Jev decision-serving family + the open recipe), 655 (model-zoo hygiene — the edited-checkpoint lens)
> **Cross-ref:** riir-infer `.issues/1216` (the untried cell, pre-registered PoC), riir-train `.issues/665` addendum (external evidence vs the two Law-4 kills)
> **Classification:** Public

---

## TL;DR

cmpnd ships `WeightSpace`, a DSPy optimizer that lets a writer LLM edit a decision student's weights directly — sandboxed turns, budgeted forward passes, a hash-checked patch over the public base, guardrails refusing embeddings/output-layer/final-norm edits and whole-tensor rescales. On Typed Decisions (400 held-out cases, paired, ±0.7–1.4 pts SE) every model improved: +1.9–16.1 pts from the edit alone, +6.5–20.0 composed with ReAnchor (threshold refit, no weights). Three findings matter to us far more than the scores:

1. **The `simulate_head` law:** allowing the writer to recompute the decision head from reads and derive the edit CLOSED-FORM is worth 5.75× over forward-only search (+16.1 vs +2.8 on d1-omni-600M; under the paper's forward-only rule the edit "added nothing to ReAnchor"). External corroboration of the modelless-unblock Path 2 — and a mapping finding: our three adjacent kills sit on DIFFERENT axes (Lane 2 = writer-arm × direct rig metric; T6 = closed-form × backbone × teacher-KL; Plan 425 = gradient × head scorer × banking77, encoder-ceiling), so the untried cell is narrow — closed-form author × typed checkpoint × typed_decisions × the head scorer surface.
2. **Proposer quality dominates and can go NEGATIVE** (writer-class swing: gpt-6.1-sol −0.6 ↔ gpt-6-astra +16.1 on the same model) — the Proposer-utility law (riir-train Issue 665 Law 4) read from outside. Caveat that keeps our gate honest: their control is a weaker WRITER, never a budget-matched random search, so their numbers alone do not adjudicate Law 4.
3. **Two-axis composition measured** — threshold refit (serving-config axis) stacks with weight edits (weights axis); on the weakest-edit model the composition carried the gain (Clef: +1.9 alone → +6.5 combined).

**Distilled for katgpt-rs (modelless, inference-time):** the decision head's SCORER block (LayerNorm → d→d GELU → d→1 — fp32 at dense loads; `Weight2D` also carries Q8/Q4, where the per-family lossy rule applies, not 'requant never arises') is the high-leverage, low-blast-radius surface: encoder frozen, and the solve is a tiny ridge / least-squares / LP fit over the post-GELU design matrix. THE CLAIM STAYS NARROW: closed-form authoring of that surface is UNTRIED in-stack. The nearest kill is riir-train Plan 425 (Bench 613): a GRADIENT fine-tune of the same scorer on banking77 saturated at the teacher's own accuracy (best holdout 0.4110 vs bar 0.80; train 0.4319 ≈ teacher 0.4312) because the ENCODER representations do not separate that suite — an encoder-ceiling kill, not a fitting-method kill; GD upper-bounds a closed-form solve on the SAME cell, so the PoC stands or dies on the cell being different (typed checkpoint × typed_decisions) and on its headroom gate. Three of the four substrates ship (`weight_edit_overlay` 980, `lp_authoring` 993, Plan 425's dump lane for the reads); the paired scoring lives reflex-side.

---

## 1. Source core findings (their numbers, their guards)

| Finding | Numbers | Our reading |
|---|---|---|
| Every decision model improved | d1-omni-600M 50.9→66.9 (+16.1); d1-2B 60.6→74.0 (+13.4); Clef-Flash 9B 68.8→70.7 (+1.9); repo: NanoJev +13.3, Strands 2B +10.4 | the class works on decision models, not just text LLMs (648's Exp-8 was Qwen3.5-0.8B on MATH-500) |
| ReAnchor composition | combined +6.5 (Clef) to +20.0 (Strands); on Strands ReAnchor ALONE (+13.7) beat the edit (+10.4) | the two-axes law measured: serving-config fit vs weight edit compose; the cheap axis can dominate |
| simulate_head (closed-form head solve from reads) | +16.1 with vs +2.8 ± 0.7 without (d1-omni re-run under the paper's rule); without it, "added nothing to ReAnchor" | THE finding: direct-read closed-form ≫ black-box search on the head surface — Path 2 with a decision-head address. CAVEAT (named): `simulate_head` is still a writer LLM using head recomputation as a TOOL; a writer-free solve is our extrapolation, and the PoC must earn it |
| Writer class swing | gpt-6.1-sol +5.8 / +12.1 / −0.6 vs gpt-6-astra +16.1 / +13.3 / +13.4 (d1-omni / NanoJev / d1-2B) | proposer capability is the dominant variable; a weak proposer can lose to NO edit |
| Guardrails | refuse embeddings / output layer / final norm; refuse whole-tensor rescales; in practice every writer edited the decision head (or d1-2B's last block) | the head is the high-leverage, low-blast-radius surface; corroborates T7's certificate direction (riir-train Issue 665) |
| Obstacle-vs-ability corroboration | Gemma 3 270M follow-up "barely moved — didn't loop, so the writer had no failure mechanism to repair" | Research 648's law, externally re-measured |
| Artifact | patch = changed entries + sha256(base, edited), bit-for-bit rebuild from the public base; patch (MBs) vs full checkpoint | our overlay law verbatim: `WeightEditArtifact` + `ProvenanceTrailer` + the instinct epoch-tag monotonic apply |
| Measurement honesty | their unoptimized scores differ from TypeSafe's leaderboard (request format + measure: program-returned answer after thresholds vs argmax); edited models belong with FITTED rows, not zero-shot | the contamination/measurement discipline (grammar-not-answers posture) + the zoo-hygiene lens (655) |
| Cost / search | $5–10 writer spend, 68–167 min per 25-pass run; GEPA search = a population of writer sessions on a Pareto frontier (perfect-parent rounds skip the writer) | hosted-writer costs are real; our routed closed-form arm needs no hosted writer by construction (a local solve — whether it earns the posture is the PoC's question, not a measured fact) |

## 2. Distillation — the three kills, their axes, and the narrow untried cell

Our stack has THREE recorded kills/negatives on adjacent cells, and they split across DIFFERENT axes — the mapping is the finding:

| Cell | Surface | Author / supervision | Kill mode |
|---|---|---|---|
| riir-train Issue 665 **Lane 2** (Bench 706, 2026-11-11) | a from-scratch drive-net fed into the leash FSM, scored on the real-boot rig | hosted-LLM writer × direct task metric | Law-4 FAIL: ≈ best-of-8 RandOpt (paired LCB95 −0.0592) — **the closest in-stack analogue of dspy-wsp's WRITER arm, and it ran the random control dspy-wsp lacks** |
| riir-train Issue 665 **T6** (rethink Bench 099, 2026-11-11) | BACKBONE MLP (fp16 ≤1B) | closed-form solve × teacher-KL pull-back | Law-4 FAIL (tied RandOpt, A−F LCB95 −0.035) + G1 FAIL (0.150 vs 0.820) |
| riir-train **Plan 425** (Bench 613, 2026-09-28) | HEAD scorer (s1+s3, 1.05M params; encoder frozen), banking77, english checkpoint | GRADIENT (AdamW) × direct labels | ENCODER-CEILING kill: holdout 0.4110 ≈ the teacher's own 0.4312 — "no readout on top can fix that" — a features-don't-separate failure, NOT a fitting-method failure |

dspy-wsp's winning cell — the DECISION HEAD surface × direct task-metric reads on production decision models — overlaps none of the three exactly: Lane 2 had the writer + direct metric but no pretrained readout; T6 had closed-form but backbone × indirect; 425 had head × direct but GRADIENT, and died on an encoder ceiling specific to banking77. **What remains untried is narrow, and the claim stays narrow: closed-form author × typed checkpoint × typed_decisions × the head scorer surface.** GD upper-bounds a closed-form solve on the SAME cell (425 is that warning), so the PoC's survival depends on the cell difference + its headroom gate, never on the solve method. Filed as riir-infer Issue 1216 (laya lane: public checkpoints, our own harness, no hosted writer in the primary arm). Law 4 STAYS the adjudicator — dspy-wsp reports no budget-matched random control, so external numbers cannot pre-clear the arm; the norm-matched RandOpt arm + the T7 guard certificate (protection-OFF must red; seeded-random-256 must harm; tampered envelope refuses loud) carry over verbatim.

## 3. Prior art

**Internal-first (grepped this session):** the WSP family corpus — 648 (parent), 659 (MEMIT ridge, Issue 990 CLOSED NEGATIVE), 664 (LP arm, Issue 993 landed), 668 (Tree-Ising discrete format-native — the third solver successor on the same repair task), 649 (probe author) — plus the decision-serving family: 562 + 573 (TypeSafe/Jev; the Jeva recipe), katgpt-rs Issue 1209's discrete-authoring arm on the 980 lane, and the kill ledger: riir-train Issue 665 (Lane 2 + T6) + Plan 425 / Bench 613 (the head-scorer fine-tune negative — the nearest kill, §2). **External:** He's `llm-honing` (decision-model editing, named by cmpnd as the earlier work). Web check 2026-11-14: the decision-model class is hot (TypeSafe Jev 2026-09-15; Strands Decider 2B 2026-10-01; Jeva arXiv:2610.18773; `convaiinnovations/laya` public on HF with typed-decisions/system-one/rlcd tags) but NO published decision-model WSP beyond cmpnd + He. Q1 not claimed.

**Family resolution (closes vocab §10's hedge):** our reflex AgentJev lane dials `malevrigns/agent-jev` (third-party, Apache-2.0, "not affiliated" — riir-reflex `harness.rs` :80) — not TypeSafe's hosted Jev, not NanoJev (`C-Tianyu/NanoJev`). dspy-wsp's five backends (d1/Liquid AI, NanoJev/C-Tianyu, d1-2B/Liquid LFM3, Clef/Cloudflare, Strands/strands-labs) include NONE of the laya family — dspy-wsp does not support laya. Same benchmark name (Typed Decisions), same wire shape (`/v1/systemone`); the leaderboard they diff against is TypeSafe's. The laya decision-head overlay surface is therefore unclaimed and ours to host.

## 4. Verdict

**Gain** — both tracks, filed not built:

- **Modelless/inference track:** Gain → riir-infer Issue 1216 (the untried cell, pre-registered: Law-4 RandOpt arm + T7 guard certificate + scorer-only guardrail + the contamination law). Proof path lives in the issue; no katgpt-core code change in this note.
- **Production track:** Gain → riir-train Issue 665 addendum (external evidence bearing on Laws 1/3/4; the Lane-2 and T6 verdicts STAND — this note records the re-open INPUT, not a re-open).
- **Serving track:** no reflex/rethink files; the composition cell (threshold-refit arm) rides Issue 1216's harness and reflex's existing cal-fit levers at run time.
- **MOAT gate (katgpt-rs):** substrate-level evidence record + routing; nothing new to implement here — the priority-#1 surface (the self-evolve decision family) is reached THROUGH Issue 1216's harness scoring on our own benchmark.
- **Named weak points:** (a) the 5.75× `simulate_head` ratio is one model, one cell, external, with no random-search control in their data, and `simulate_head` is a writer-with-a-tool — the writer-free solve this note routes is an extrapolation the PoC must earn; (b) GD upper-bounds closed-form on the same cell (Plan 425), so survival rides the typed/typed_decisions cell difference + the headroom gate; (c) external numbers are quote-pinned to fetched blog/README text, not a clone.

## 5. Fusion

**The four-substrate composition (this note's one new capability):** laya head reads (Plan 425's dump lane — `encode_question` + `dump_encoder_states` + scorer-tensors-as-data, Bench 613's companion landing; `head.rs::reads_of` alone returns per-option logits + CLS sync, NOT the design matrix — the solve reconstructs the post-GELU s1 matrix from captured marker rows + scorer weights exactly as 613's train-side reimplementation did) × closed-form author (`lp_authoring` min-L1 PDHG or `quant_error_lora` rank-r ridge) × `weight_edit_overlay` apply (`EditForm::DirectionSubtraction` column-masked rank-1, `forward_delta_into` dequant-domain apply) × our typed_decisions paired harness with the reflex threshold-refit composition arm. None of the four alone gives a WSP-authored decision head with paired proof on our own benchmark.

**GEPA population search** ↔ `gepa_reflective` (katgpt-pruners, Plan 164, default-on) + `ruliology_search`: the config-evolution machinery could drive overlay-candidate population search — ORDERED after the single-shot cell proves signal (single-shot → population; never skip the Law-4 arm).

## 6. Follow-ups

- riir-infer `.issues/1216` — the head-surface PoC (unclaimed, queueable; local laya checkpoints + local solve; light enough for m3, rig-queueable for the paired cells).
- riir-train `.issues/665` — addendum appended (external evidence vs the two Law-4 kills; T7 certificate corroboration).
- `vocab.md` §10 — the "not yet distilled" line resolved → this note; the family row resolved (identity refuted); the GEPA row gains `gepa_reflective`.
